# Guide Vocals

Available on `main`; originally developed on `guide-vocal-experiment`.

For optional correction of the original's detected pitch, see
[Correct brief octave jumps](pitch-correction.md).

Use a second take to influence pronunciation while the original supplies the
performance and the selected RVC model supplies the output voice. With octave
correction enabled, the guide can also identify the source's intended register
without copying its exact tuning. No retraining or new model downloads are
required when the normal RVC models are installed. Full-length tracks are
supported; there is no 30-second limit.

Pronunciation and accent transfer depend on the guide and the selected model.
Content features also carry some delivery information. Strong guidance can change
expression or create artifacts. Lyrics-driven correction and explicit transfer
of the guide's exact pitch or vibrato are not implemented.

## Web UI

1. Start the Web UI normally, select the voice/index, and enter the original
   audio path and your usual conversion settings.
2. Use the ordinary **Convert** button to get a baseline.
3. Open **Guide Vocals** in the single-conversion tab. Upload/record a
   guide or enter its local audio path. Use isolated vocals with the same lyrics
   and verse order as the original. A roughly sung, monotone guide is usable;
   clear pronunciation and comparable phrasing matter more than accurate pitch.
4. Start with the default **retrieval** mode and strength **0.5**, then **Convert with guide**.
   The guided result has its own player so the ordinary output stays available.
5. Compare 0.35, 0.7 and 1.0 to hear how much original articulation to retain.
   Try **content** mode when you want stronger pronunciation changes.
   Strength 0 uses ordinary conversion and does not load the guide.

**content** blends aligned guide content features into the original before
retrieval and synthesis. **retrieval** blends the search query, while retaining
the original features for the non-retrieved part of the output. Retrieval mode
requires a usable index and an index rate greater than zero. Use the same index
rate as the baseline; guidance strength and index rate are separate controls.
Retrieval is the default in the Web UI, CLI, and Python API.

The selected F0 extractor, pitch shift/range or supplied F0 curve, output sample
rate, and loudness-envelope settings continue to apply. A pitch-enabled model is
required for guided conversion. Existing consonant/breath protection uses the
original content and voicing; strong protection can therefore reduce changes to
unvoiced consonants. A conservative energy gate suppresses guidance in silence.

When **Correct octave / overtone errors** is enabled, RMVPE reads the aligned
guide only to choose between adjacent octaves of the selected extractor's result
and to shape missing-F0 bridges. It can absorb a brief unstable detector recovery
into the bridge when the source settles again within 80 ms. The source's tuning
and ordinary intervals stay intact. The guide's prevailing octave is normalized
automatically, and its start/end region limits pitch guidance as well as
pronunciation guidance.

## Timing and selective correction

**auto** estimates a full-track monotone alignment from content features, then
refines it in small windows. Repeated verses, long instrumental gaps, different
arrangements and ambiguous vowels can still confuse the alignment. The displayed
similarity is a diagnostic, not a calibrated confidence or pronunciation score.

**linear** matches the total duration of the guide to the original. Use it when
the takes are already in sync, or differ by an approximately constant tempo.

Timing anchors override either automatic or linear alignment with an explicit
piecewise linear map. Enter one pair per line, using seconds from the beginning
of each complete file:

```text
12.5,13.2
18.0,19.1
42.0,44.0
```

This says the original's 12.5-second point corresponds to 13.2 seconds in the
guide, and so on. Times must increase strictly in both recordings. File start/end
anchors are supplied automatically unless explicitly entered. Set anchors around
a problem word's onset and vowel transition if auto alignment is wrong. There is
no automatic lyric/phoneme labeling in this version.

**Apply guide from/until source time** limits the correction to one interval of
the original. The default 0/0 covers the whole track. Both complete tracks are
still aligned; the guide is not interpreted as a standalone snippet for that
interval. Edges fade over approximately 40 ms. To guide a small region with a
short separate recording, first make corresponding clips or prepare a guide
track with matching structure.

## Reproducible command-line comparison

Use the same Python environment as the existing app (`~/.venv/bin/python` in
this checkout). Substitute your actual file/model/index paths:

```bash
~/.venv/bin/python tools/infer_cli.py \
  --input_path /path/to/original.wav \
  --guide_path /path/to/guide.wav \
  --model_name 'Wes.pth' \
  --index_path '/path/to/voice.index' \
  --f0method rmvpe --index_rate 0.75 --rms_mix_rate 0.25 \
  --opt_path /path/to/comparison.wav --compare --seed 67
```

This writes `comparison_baseline.wav`, `comparison_retrieval_050.wav` (when an
index is supplied), and content variants at 0.35, 0.7 and 1.0. All use the same
synthesis seed. `comparison.guide.json` records settings and frame timing maps.
Separate Web UI conversions do not reset the synthesis seed, so there can be
small random differences even with identical settings.

For one guided render, omit `--compare`; retrieval mode is the default. Use
`--guide_mode content` to select content mode, and `--guide_strength 0.7` to
adjust strength. Other optional controls are `--guide_alignment`,
`--guide_anchors_path` (a text file), `--guide_start`, and `--guide_end`.
Existing CLI calls without a guide retain their previous behavior.

## Implementation and checks

Guide/source features are extracted in 20-second blocks with one second of
context on each side. Only context is discarded, so feature frames stay on one
continuous 20 ms timeline. These processing blocks do not limit track length.
Guide alignment uses a bounded coarse matrix followed by bounded local DTW
problems; it never allocates a quadratic full-song matrix. Feature storage still
grows with track length, and guided conversion takes longer than ordinary RVC.

The aligned guide is sampled on each synthesis chunk's global timeline, including
half-frame chunk starts and context padding. Optional register correction uses
the same timing map on the 10 ms pitch grid. Zero-distance index matches are
handled without NaNs.

Run regression tests with:

```bash
~/.venv/bin/python -m unittest discover -s tests -v
```

Listening to a genuine alternate guide is still necessary to judge vowel
correction and delivery; timing-altered copies and synthetic feature tests only
verify mechanics.

Initial validation on this checkout: nine regression tests passed; both guidance
modes rendered with the existing Wes model on CPU; a 48-second original and
52.8-second timing-altered guide completed across multiple extraction/synthesis
chunks. Under a fixed seed, ordinary output matched the pre-branch implementation
bit for bit, and strength zero ignored a nonexistent guide path. The full UI
built with separate output players and the existing API signature intact. The
CLI comparison produced all five WAV variants and its timing-map manifest.
