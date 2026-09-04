# Correct octave / overtone errors

Enable **Correct octave / overtone errors**, next to the pitch-range field, to
reduce pitch-doubling and halving errors. It works in ordinary, Guide Vocals,
and batch conversion, and stays off by default for comparison and splicing.

The correction considers the original detected pitch and alternatives one or
two octaves above/below it. It proposes a continuous path across each phrase,
then checks proposed shifts against stable preceding notes and the source
waveform. It favors the original detection when alternatives are ambiguous. Candidate
corrections stay within the detector's 50–1100 Hz range before transposition.

Unlike the earlier brief-jump correction:

- There is **no 250 ms limit** and no requirement to return to the starting note.
- The surrounding notes can differ; errors can span changing notes, slides, and
  phrase endings.
- Short unvoiced gaps (up to 150 ms) can connect context, without filling those
  gaps with pitch. Longer gaps reset the phrase's register reference.
- Waveform periodicity helps distinguish a strong overtone from a fundamental.
  It is evidence, not a guarantee: multiples of the true period can also fit.

## Protecting ordinary melodic movement

A low syllable ending or noisy attack must not pull the next valid note down an
octave. Proposed shifts now need a stable preceding note: approximately 120 ms
within a two-semitone span, found within the preceding 350 ms. Reflected padding,
unvoiced gaps, and unaccepted proposals cannot establish this reference.

Ordinary intervals through a perfect fifth are allowed without a continuity
penalty. An octave shift must improve the interval substantially, or have clear
support from the source waveform. When audio is supplied, the reference and
proposed pitch must also provide sufficient periodicity evidence together.
Weak or ambiguous waveform evidence leaves the detected pitch unchanged.

The correction checks again at a new stable note or a consonant gap of 30 ms
or more. A brief detector recovery of up to 40 ms inside an already supported
slide can reconnect to that corrected trajectory. This avoids introducing a
new jump when an octave error momentarily disappears during a descending note.
Phrase openings retain the detector's register until stable context exists.

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

Measured regression fixtures contain pitch and periodicity values, without
waveform audio. They cover the original octave plateau and descending-tail
errors, plus the valid melody after low syllable transitions and the ambiguous
passage that the earlier correction incorrectly changed.
