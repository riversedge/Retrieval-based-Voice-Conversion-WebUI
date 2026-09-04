# Correct octave / overtone errors

Enable **Correct octave / overtone errors**, next to the pitch-range field, to
reduce pitch-doubling and halving errors. It works in ordinary, Guide Vocals,
and batch conversion, and stays off by default for comparison and splicing.

The correction considers the original detected pitch and alternatives one or
two octaves above/below it. It chooses a continuous path across each phrase,
using the source waveform's periodicity as supporting evidence and favoring the
original detection when the alternatives are similarly plausible. Candidate
corrections stay within the detector's 50–1100 Hz range before transposition.

Unlike the earlier brief-jump correction:

- There is **no 250 ms limit** and no requirement to return to the starting note.
- The surrounding notes can differ; errors can span changing notes, slides, and
  phrase endings.
- Short unvoiced gaps (up to 150 ms) can connect context, without filling those
  gaps with pitch. Longer gaps reset the phrase's register reference.
- Waveform periodicity helps distinguish a strong overtone from a fundamental.
  It is evidence, not a guarantee: multiples of the true period can also fit.

Every change is an exact octave shift. Vibrato, detuning, and slides within the
chosen octave remain intact. “Between different notes” does not mean snapping
notes to a key or correcting arbitrary intervals: this feature selects octaves
of the detected notes. It does not repair every harmony or synthesis artifact.

The option deliberately favors continuity over large abrupt leaps and **may
change intentional melody jumps**, including sustained ones. Run with it on and
off when choosing takes. A whole phrase detected in the wrong octave can still
be ambiguous without a reliable register reference.

No pitch range is required. Correction runs after pitch extraction and before
transposition and the existing range adjustment. For example, a mistaken 900 Hz
detection can first become 450 Hz, then become 225 Hz with a -12 semitone shift.
In Guide Vocals the source performance provides this pitch and waveform evidence;
the guide continues to control pronunciation.

The conversion information reports adjusted frames and seconds. Reflected model
padding is excluded from both pitch context and waveform evidence. Supplied
external F0 curves bypass correction, and models without pitch conditioning
report that it was skipped. Turning the option off bypasses the new analysis.
The source-waveform analysis runs in bounded batches; full-length tracks are
supported without a duration cap.

## CLI and Python

The existing `--correct_octave_errors` flag and `correct_octave_errors=True`
Python argument remain unchanged. CLI `--compare` applies the chosen correction
setting to every comparison variant. Existing API argument positions remain
unchanged. The old `correct_brief_octave_jumps` helper remains as a compatibility
alias to the new continuity behavior, without waveform evidence.

## Verification

Run `~/.venv/bin/python -m unittest discover -s tests -v` in this checkout.
Tests cover changing notes, descending phrase endings, sustained errors,
vibrato, ordinary melodic intervals, slides, short and long gaps, waveform
evidence, transposition, external F0 priority, and the disabled path. Guide
Vocals regression tests remain part of the suite.
