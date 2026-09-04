# Correct brief octave jumps

Enable **Correct brief octave jumps**, next to the pitch-range field, to reduce
short pitch-doubling or halving errors. The option is available for ordinary,
Guide Vocals, and batch conversion. It is off by default.

The detector may briefly report 440 Hz while surrounding notes are near 220 Hz,
or drop to 110 Hz. When voiced context on both sides agrees, the correction moves
the suspect section by exactly one octave. It retains the section's vibrato and
small pitch variations instead of replacing it with a flat note.

The initial conservative settings are:

- A suspect excursion lasts at most **250 ms**.
- Its entry and exit are approximately one octave, within two semitones.
- There is at least **50 ms of voiced context on both sides**, examining up to
  120 ms per side. Neighboring pitches must be stable and in a similar register.
- After shifting by one octave, the section must fit the neighboring pitches.

No pitch range is required. The correction runs after detection, before the
existing transposition and pitch-range adjustment. Those controls still work
normally. In Guide Vocals, it corrects the original performance's detected F0;
the guide continues to control pronunciation.

The conversion information reports how many frames/seconds were adjusted.
Reflected model padding does not count as musical context. Unvoiced gaps are not
filled, and supplied external F0 curves bypass the correction. Models without
pitch conditioning report that the correction was skipped.

This is a continuity heuristic, not a harmony recognizer or automatic tuning to
a scale. It deliberately leaves sustained wrong-register passages, gradual
slides, non-octave errors, and ambiguous phrase boundaries alone. A genuinely
intended short octave leap can look identical to a detector error and may be
changed; disable the option if it alters the melody.

## CLI and Python

Add `--correct_octave_errors` to an existing `tools/infer_cli.py` command. It also
works with `--guide_path` and `--compare`; comparison runs all use the same octave
correction setting.

Python callers can pass `correct_octave_errors=True` to `VC.vc_single` or
`VC.vc_multi`. Existing positional arguments remain in place. The Web UI APIs
append the checkbox after their existing inputs.

## Verification

Run `~/.venv/bin/python -m unittest discover -s tests -v` in this checkout.
Coverage includes doubled/halved pitch with vibrato, adjacent bursts, sustained
octaves, slides, non-octave notes, breaths, phrase boundaries, disabled behavior,
manual F0 priority, and interaction with transposition and pitch range.
