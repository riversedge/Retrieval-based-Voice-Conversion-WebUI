# Correct octave / overtone errors

Enable **Correct octave / overtone errors**, next to the pitch-range field, to
reduce pitch-doubling and halving errors. It works in ordinary, Guide Vocals,
and batch conversion, and stays off by default for comparison and splicing.

The selected pitch extractor remains authoritative. The correction does not run
FCPE or substitute estimates from a second detector. It can choose the adjacent
octave of a detected pitch and can bridge missing estimates while source audio
continues. Sustained source silence remains unvoiced.

## With Guide Vocals

When an aligned guide is supplied, RMVPE analyzes the guide as register evidence.
The guide's prevailing whole-octave difference from the source is normalized, so
a guide sung an octave above or below the original still works. At each voiced
source frame, the correction considers the source estimate, half that frequency,
and twice that frequency. It changes the source only when an adjacent octave is
more than six semitones closer to the guide's register.

After a detector dropout of at least 20 ms, the first recovered estimates may be
unstable. The correction can extend the bridge by up to 80 ms when those estimates
start at least 4.5 semitones from the guide and then settle within three semitones.
This is limited to detector recovery; an uninterrupted flat or off-key source note
is retained.

This does not copy the guide singer's note tuning, vibrato, or arbitrary melodic
intervals. A slightly flat source note remains slightly flat. The guide only
chooses an octave for source estimates. During a detector dropout, the guide can
supply relative pitch movement; the bridge is offset at both ends to meet the
source-derived pitches. If the guide is unvoiced or incomplete there, the bridge
interpolates between the surrounding source pitches instead.

Guide pitch follows the same automatic, linear, or anchored timing map used for
pronunciation. The guide start/end region also limits its register evidence.
Guide pronunciation strength may be zero while its pitch is used by this option.

## Without a guide

The correction uses stable notes and source-waveform periodicity to maintain a
continuous register through an audible phrase. Only one-octave alternatives are
considered. Normal melodic intervals, slides, vibrato, and detuning within the
chosen octave remain intact. Ambiguous changes retain the selected extractor's
pitch.

Missing detector estimates are interpolated in log-frequency between surrounding
source estimates. At an audible file boundary, the nearest available pitch is
held. A gap that reaches sustained source silence is left unvoiced, and silence
resets the phrase reference.

The option favors register continuity over abrupt mid-phrase octave leaps and may
change an intentional leap. Run with it off when the leap is part of the intended
performance. No pitch range is required. Correction runs after pitch extraction
and before transposition and range adjustment. Supplied external F0 curves bypass
correction, and models without pitch conditioning report that it was skipped.
Full-length tracks are supported without a duration cap.

## CLI and Python

The existing `--correct_octave_errors` flag and `correct_octave_errors=True`
Python argument remain unchanged. Existing API argument positions also remain
unchanged.

Run the pitch and guide regression tests with:

```bash
~/.venv/bin/python -m unittest discover -s tests -p 'test_pitch*.py'
~/.venv/bin/python -m unittest discover -s tests -p 'test_guide.py'
```
