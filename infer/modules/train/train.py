import os
import sys
import logging
import traceback
from contextlib import nullcontext

logger = logging.getLogger(__name__)

now_dir = os.getcwd()
sys.path.append(os.path.join(now_dir))

import datetime

from infer.lib.train import utils
from infer.lib.device import get_device, device_str
from infer.lib.torch_load_compat import torch_load_compat

hps = utils.get_hparams()
os.environ["CUDA_VISIBLE_DEVICES"] = hps.gpus.replace("-", ",")
n_gpus = len(hps.gpus.split("-"))
from random import randint, shuffle

import torch

try:
    import intel_extension_for_pytorch as ipex  # pylint: disable=import-error, unused-import

    if torch.xpu.is_available():
        from infer.modules.ipex import ipex_init
        from infer.modules.ipex.gradscaler import gradscaler_init

        GradScaler = gradscaler_init()
        ipex_init()
    else:
        from torch.cuda.amp import GradScaler
except Exception:
    from torch.cuda.amp import GradScaler

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
from time import sleep
from time import time as ttime

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from infer.lib.infer_pack import commons
from infer.lib.train.data_utils import (
    DistributedBucketSampler,
    TextAudioCollate,
    TextAudioCollateMultiNSFsid,
    TextAudioLoader,
    TextAudioLoaderMultiNSFsid,
)

if hps.version == "v1":
    from infer.lib.infer_pack.models import MultiPeriodDiscriminator
    from infer.lib.infer_pack.models import SynthesizerTrnMs256NSFsid as RVC_Model_f0
    from infer.lib.infer_pack.models import (
        SynthesizerTrnMs256NSFsid_nono as RVC_Model_nof0,
    )
else:
    from infer.lib.infer_pack.models import (
        SynthesizerTrnMs768NSFsid as RVC_Model_f0,
        SynthesizerTrnMs768NSFsid_nono as RVC_Model_nof0,
        MultiPeriodDiscriminatorV2 as MultiPeriodDiscriminator,
    )

from infer.lib.train.losses import (
    discriminator_loss,
    feature_loss,
    generator_loss,
    kl_loss,
)
from infer.lib.train.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from infer.lib.train.process_ckpt import savee, build_model_config

global_step = 0

# --- PAD TRACER (debug) ---
_PAD_TRACER_ENABLED = os.environ.get("RVC_PAD_TRACE", "0") == "1"
_PAD_TRACER_FIRED = False
_orig_pad = F.pad


def _pad_tracer(x, pad, mode="constant", value=0):
    global _PAD_TRACER_FIRED
    try:
        if (
            _PAD_TRACER_ENABLED
            and not _PAD_TRACER_FIRED
            and hasattr(x, "is_mps")
            and x.is_mps
            and getattr(x, "dim", lambda: 0)() > 3
            and mode == "constant"
        ):
            _PAD_TRACER_FIRED = True
            print("\n=== RVC_PAD_TRACE HIT ===")
            print(f"shape={tuple(x.shape)} dtype={x.dtype} device={x.device}")
            print(f"pad={pad} mode={mode} value={value}")
            print("traceback (most recent call last):")
            print("".join(traceback.format_stack(limit=20)))
            print("=== END RVC_PAD_TRACE ===\n")
    except Exception as exc:
        print(f"[RVC_PAD_TRACE] tracer error: {exc}")
    return _orig_pad(x, pad, mode, value)


if _PAD_TRACER_ENABLED:
    F.pad = _pad_tracer
# --- END PAD TRACER ---


class NullScaler:
    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        return None

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        return None


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value not in {"0", "false", "False", "no", "NO"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


class EpochRecorder:
    def __init__(self):
        self.last_time = ttime()

    def record(self):
        now_time = ttime()
        elapsed_time = now_time - self.last_time
        self.last_time = now_time
        elapsed_time_str = str(datetime.timedelta(seconds=elapsed_time))
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"[{current_time}] | ({elapsed_time_str})"


def main():
    device = get_device()
    ddp_enabled = _env_flag("RVC_DDP", device.type == "cuda")
    if device.type != "cuda":
        ddp_enabled = _env_flag("RVC_DDP", False)
    if device.type == "cuda" and ddp_enabled:
        n_gpus = torch.cuda.device_count()
    else:
        n_gpus = 1
    if n_gpus < 1:
        print("NO GPU DETECTED: falling back to CPU - this may take a while")
        n_gpus = 1
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(randint(20000, 55555))
    children = []
    logger = utils.get_logger(hps.model_dir)
    if device.type == "mps" and _env_flag("RVC_DDP", False):
        logger.warning(
            "MPS backend is single-device; disabling DDP and running a single process."
        )
    logger.info("Using device: %s (DDP=%s)", device_str(device), ddp_enabled)
    for i in range(n_gpus):
        subproc = mp.Process(
            target=run,
            args=(i, n_gpus, hps, logger, ddp_enabled, device),
        )
        children.append(subproc)
        subproc.start()

    for i in range(n_gpus):
        children[i].join()


def run(rank, n_gpus, hps, logger: logging.Logger, ddp_enabled, device):
    global global_step
    device_type = device.type if isinstance(device, torch.device) else str(device)
    if device_type == "cuda":
        device = torch.device(f"cuda:{rank}") if ddp_enabled else device
    if rank == 0:
        # logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        # utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    if ddp_enabled:
        dist.init_process_group(
            backend="gloo", init_method="env://", world_size=n_gpus, rank=rank
        )
    torch.manual_seed(hps.train.seed)
    if device_type == "cuda":
        torch.cuda.set_device(rank)

    if device_type == "mps":
        default_threads = min(4, os.cpu_count() or 1)
        num_threads = _env_int("RVC_NUM_THREADS", default_threads)
        num_interop = _env_int("RVC_NUM_INTEROP_THREADS", 1)
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(num_interop)

    fp16_run = _env_flag("RVC_FP16", hps.train.fp16_run)
    amp_enabled = _env_flag("RVC_AMP", fp16_run)
    if device_type != "cuda":
        fp16_run = _env_flag("RVC_FP16", False)
        amp_enabled = _env_flag("RVC_AMP", False)
    if not amp_enabled:
        fp16_run = False
    hps.train.fp16_run = fp16_run

    if hps.if_f0 == 1:
        train_dataset = TextAudioLoaderMultiNSFsid(hps.data.training_files, hps.data)
    else:
        train_dataset = TextAudioLoader(hps.data.training_files, hps.data)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size * n_gpus,
        # [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200,1400],  # 16s
        [100, 200, 300, 400, 500, 600, 700, 800, 900],  # 16s
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )
    # It is possible that dataloader's workers are out of shared memory. Please try to raise your shared memory limit.
    # num_workers=8 -> num_workers=4
    if hps.if_f0 == 1:
        collate_fn = TextAudioCollateMultiNSFsid()
    else:
        collate_fn = TextAudioCollate()
    pin_memory_default = device_type == "cuda"
    if device_type == "mps":
        pin_memory_default = False
    num_workers = _env_int("RVC_NUM_WORKERS", 4 if device_type == "cuda" else 2)
    pin_memory = _env_flag("RVC_PIN_MEMORY", pin_memory_default)
    persistent_workers = num_workers > 0
    prefetch_factor = 8 if num_workers > 0 else None
    train_loader = DataLoader(
        train_dataset,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    if hps.if_f0 == 1:
        net_g = RVC_Model_f0(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model,
            is_half=fp16_run,
            sr=hps.sample_rate,
        )
    else:
        net_g = RVC_Model_nof0(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model,
            is_half=hps.train.fp16_run,
        )
    net_g = net_g.to(device)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm)
    net_d = net_d.to(device)
    if rank == 0:
        logger.info(
            "Training device: %s (net_g params: %s)",
            device_str(device),
            next(net_g.parameters()).device,
        )
        print("MODEL DEVICE:", next(net_g.parameters()).device)
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    # net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
    # net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=True)
    if ddp_enabled:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            pass
        elif device_type == "cuda":
            net_g = DDP(net_g, device_ids=[rank])
            net_d = DDP(net_d, device_ids=[rank])
        else:
            net_g = DDP(net_g)
            net_d = DDP(net_d)

    try:  # 如果能加载自动resume
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d
        )  # D多半加载没事
        if rank == 0:
            logger.info("loaded D")
        # _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g,load_opt=0)
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g
        )
        global_step = (epoch_str - 1) * len(train_loader)
        # epoch_str = 1
        # global_step = 0
    except:  # 如果首次不能加载，加载pretrain
        # traceback.print_exc()
        epoch_str = 1
        global_step = 0
        if hps.pretrainG != "":
            if rank == 0:
                logger.info("loaded pretrained %s" % (hps.pretrainG))
            if hasattr(net_g, "module"):
                logger.info(
                    net_g.module.load_state_dict(
                        torch_load_compat(hps.pretrainG, map_location="cpu")[
                            "model"
                        ]
                    )
                )  ##测试不加载优化器
            else:
                logger.info(
                    net_g.load_state_dict(
                        torch_load_compat(hps.pretrainG, map_location="cpu")[
                            "model"
                        ]
                    )
                )  ##测试不加载优化器
        if hps.pretrainD != "":
            if rank == 0:
                logger.info("loaded pretrained %s" % (hps.pretrainD))
            if hasattr(net_d, "module"):
                logger.info(
                    net_d.module.load_state_dict(
                        torch_load_compat(hps.pretrainD, map_location="cpu")[
                            "model"
                        ]
                    )
                )
            else:
                logger.info(
                    net_d.load_state_dict(
                        torch_load_compat(hps.pretrainD, map_location="cpu")[
                            "model"
                        ]
                    )
                )

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )

    scaler = (
        GradScaler(enabled=amp_enabled and device_type == "cuda")
        if amp_enabled and device_type == "cuda"
        else NullScaler()
    )

    cache = []
    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                [train_loader, None],
                logger,
                [writer, writer_eval],
                cache,
                device,
                amp_enabled,
            )
        else:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                [train_loader, None],
                None,
                None,
                cache,
                device,
                amp_enabled,
            )
        scheduler_g.step()
        scheduler_d.step()


def train_and_evaluate(
    rank,
    epoch,
    hps,
    nets,
    optims,
    schedulers,
    scaler,
    loaders,
    logger,
    writers,
    cache,
    device,
    amp_enabled,
):
    net_g, net_d = nets
    optim_g, optim_d = optims
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()

    device_type = device.type if isinstance(device, torch.device) else str(device)

    def autocast_context(enabled: bool):
        if not enabled:
            return nullcontext()
        if device_type == "cuda":
            return torch.cuda.amp.autocast()
        return torch.autocast(device_type)

    # Prepare data iterator
    if hps.if_cache_data_in_gpu == True:
        # Use Cache
        data_iterator = cache
        if cache == []:
            # Make new cache
            for batch_idx, info in enumerate(train_loader):
                # Unpack
                if hps.if_f0 == 1:
                    (
                        phone,
                        phone_lengths,
                        pitch,
                        pitchf,
                        spec,
                        spec_lengths,
                        wave,
                        wave_lengths,
                        sid,
                    ) = info
                else:
                    (
                        phone,
                        phone_lengths,
                        spec,
                        spec_lengths,
                        wave,
                        wave_lengths,
                        sid,
                    ) = info
                # Load on CUDA
                if device_type == "cuda":
                    phone = phone.to(device, non_blocking=True)
                    phone_lengths = phone_lengths.to(device, non_blocking=True)
                    if hps.if_f0 == 1:
                        pitch = pitch.to(device, non_blocking=True)
                        pitchf = pitchf.to(device, non_blocking=True)
                    sid = sid.to(device, non_blocking=True)
                    spec = spec.to(device, non_blocking=True)
                    spec_lengths = spec_lengths.to(device, non_blocking=True)
                    wave = wave.to(device, non_blocking=True)
                    wave_lengths = wave_lengths.to(device, non_blocking=True)
                else:
                    phone = phone.to(device)
                    phone_lengths = phone_lengths.to(device)
                    if hps.if_f0 == 1:
                        pitch = pitch.to(device)
                        pitchf = pitchf.to(device)
                    sid = sid.to(device)
                    spec = spec.to(device)
                    spec_lengths = spec_lengths.to(device)
                    wave = wave.to(device)
                    wave_lengths = wave_lengths.to(device)
                # Cache on list
                if hps.if_f0 == 1:
                    cache.append(
                        (
                            batch_idx,
                            (
                                phone,
                                phone_lengths,
                                pitch,
                                pitchf,
                                spec,
                                spec_lengths,
                                wave,
                                wave_lengths,
                                sid,
                            ),
                        )
                    )
                else:
                    cache.append(
                        (
                            batch_idx,
                            (
                                phone,
                                phone_lengths,
                                spec,
                                spec_lengths,
                                wave,
                                wave_lengths,
                                sid,
                            ),
                        )
                    )
        else:
            # Load shuffled cache
            shuffle(cache)
    else:
        # Loader
        data_iterator = enumerate(train_loader)

    # Run steps
    epoch_recorder = EpochRecorder()
    batch_device_logged = False
    for batch_idx, info in data_iterator:
        # Data
        ## Unpack
        if hps.if_f0 == 1:
            (
                phone,
                phone_lengths,
                pitch,
                pitchf,
                spec,
                spec_lengths,
                wave,
                wave_lengths,
                sid,
            ) = info
        else:
            phone, phone_lengths, spec, spec_lengths, wave, wave_lengths, sid = info
        ## Load on CUDA
        if hps.if_cache_data_in_gpu == False:
            if device_type == "cuda":
                phone = phone.to(device, non_blocking=True)
                phone_lengths = phone_lengths.to(device, non_blocking=True)
            else:
                phone = phone.to(device)
                phone_lengths = phone_lengths.to(device)
            if hps.if_f0 == 1:
                if device_type == "cuda":
                    pitch = pitch.to(device, non_blocking=True)
                    pitchf = pitchf.to(device, non_blocking=True)
                else:
                    pitch = pitch.to(device)
                    pitchf = pitchf.to(device)
            if device_type == "cuda":
                sid = sid.to(device, non_blocking=True)
                spec = spec.to(device, non_blocking=True)
                spec_lengths = spec_lengths.to(device, non_blocking=True)
                wave = wave.to(device, non_blocking=True)
            else:
                sid = sid.to(device)
                spec = spec.to(device)
                spec_lengths = spec_lengths.to(device)
                wave = wave.to(device)

        if not batch_device_logged:
            print("BATCH DEVICE:", wave.device)
            batch_device_logged = True
            # wave_lengths = wave_lengths.cuda(rank, non_blocking=True)

        # Calculate
        with autocast_context(amp_enabled):
            if hps.if_f0 == 1:
                (
                    y_hat,
                    ids_slice,
                    x_mask,
                    z_mask,
                    (z, z_p, m_p, logs_p, m_q, logs_q),
                ) = net_g(phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid)
            else:
                (
                    y_hat,
                    ids_slice,
                    x_mask,
                    z_mask,
                    (z, z_p, m_p, logs_p, m_q, logs_q),
                ) = net_g(phone, phone_lengths, spec, spec_lengths, sid)
            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            y_mel = commons.slice_segments(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length
            )
            with autocast_context(False):
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.float().squeeze(1),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
            if hps.train.fp16_run == True and device_type == "cuda":
                y_hat_mel = y_hat_mel.half()
            wave = commons.slice_segments(
                wave, ids_slice * hps.data.hop_length, hps.train.segment_size
            )  # slice

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
            with autocast_context(False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g
                )
        optim_d.zero_grad()
        scaler.scale(loss_disc).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with autocast_context(amp_enabled):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
            with autocast_context(False):
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]["lr"]
                logger.info(
                    "Train Epoch: {} [{:.0f}%]".format(
                        epoch, 100.0 * batch_idx / len(train_loader)
                    )
                )
                # Amor For Tensorboard display
                if loss_mel > 75:
                    loss_mel = 75
                if loss_kl > 9:
                    loss_kl = 9

                logger.info([global_step, lr])
                logger.info(
                    f"loss_disc={loss_disc:.3f}, loss_gen={loss_gen:.3f}, loss_fm={loss_fm:.3f},loss_mel={loss_mel:.3f}, loss_kl={loss_kl:.3f}"
                )
                scalar_dict = {
                    "loss/g/total": loss_gen_all,
                    "loss/d/total": loss_disc,
                    "learning_rate": lr,
                    "grad_norm_d": grad_norm_d,
                    "grad_norm_g": grad_norm_g,
                }
                scalar_dict.update(
                    {
                        "loss/g/fm": loss_fm,
                        "loss/g/mel": loss_mel,
                        "loss/g/kl": loss_kl,
                    }
                )

                scalar_dict.update(
                    {"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)}
                )
                scalar_dict.update(
                    {"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)}
                )
                scalar_dict.update(
                    {"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)}
                )
                image_dict = {}
                if _env_flag("RVC_TB_IMAGES", True):
                    image_dict = {
                        "slice/mel_org": utils.plot_spectrogram_to_numpy(
                            y_mel[0].data.cpu().numpy()
                        ),
                        "slice/mel_gen": utils.plot_spectrogram_to_numpy(
                            y_hat_mel[0].data.cpu().numpy()
                        ),
                        "all/mel": utils.plot_spectrogram_to_numpy(
                            mel[0].data.cpu().numpy()
                        ),
                    }
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict,
                )
        global_step += 1
    # /Run steps

    if epoch % hps.save_every_epoch == 0 and rank == 0:
        checkpoint_metadata = {
            "config": build_model_config(hps),
            "sr": hps.data.sampling_rate,
            "f0": hps.if_f0,
            "version": hps.version,
            "info": f"{epoch}epoch",
        }
        if hps.if_latest == 0:
            utils.save_checkpoint(
                net_g,
                optim_g,
                hps.train.learning_rate,
                epoch,
                os.path.join(hps.model_dir, "G_{}.pth".format(global_step)),
                metadata=checkpoint_metadata,
            )
            utils.save_checkpoint(
                net_d,
                optim_d,
                hps.train.learning_rate,
                epoch,
                os.path.join(hps.model_dir, "D_{}.pth".format(global_step)),
            )
        else:
            utils.save_checkpoint(
                net_g,
                optim_g,
                hps.train.learning_rate,
                epoch,
                os.path.join(hps.model_dir, "G_{}.pth".format(2333333)),
                metadata=checkpoint_metadata,
            )
            utils.save_checkpoint(
                net_d,
                optim_d,
                hps.train.learning_rate,
                epoch,
                os.path.join(hps.model_dir, "D_{}.pth".format(2333333)),
            )
        if rank == 0 and hps.save_every_weights == "1":
            if hasattr(net_g, "module"):
                ckpt = net_g.module.state_dict()
            else:
                ckpt = net_g.state_dict()
            logger.info(
                "saving ckpt %s_e%s:%s"
                % (
                    hps.name,
                    epoch,
                    savee(
                        ckpt,
                        hps.sample_rate,
                        hps.if_f0,
                        hps.name + "_e%s_s%s" % (epoch, global_step),
                        epoch,
                        hps.version,
                        hps,
                    ),
                )
            )

    if rank == 0:
        logger.info("====> Epoch: {} {}".format(epoch, epoch_recorder.record()))
    if epoch >= hps.total_epoch and rank == 0:
        logger.info("Training is done. The program is closed.")

        if hasattr(net_g, "module"):
            ckpt = net_g.module.state_dict()
        else:
            ckpt = net_g.state_dict()
        logger.info(
            "saving final ckpt:%s"
            % (
                savee(
                    ckpt, hps.sample_rate, hps.if_f0, hps.name, epoch, hps.version, hps
                )
            )
        )
        sleep(1)
        os._exit(2333333)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
