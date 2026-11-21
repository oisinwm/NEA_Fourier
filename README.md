2018/2019 AQA A-Level NEA Coursework

The goal of this project was to develop a Python application capable of converting an audio recording (encoded as a .wav file) into a Midi (.midi) equivilent. This was somewhat sucsessful for more simple melodies, though ran into serious efficiancy problems when faced with longer or more complitcated examples. The correctness also suffers quite a long when presented with chords, however this could not be ironed out due to the afforementioned efficiency issues. The main cause of these efficiency issues come from the self implementation of an FFT in order to gain more complexity marks, so without a significantly better understanding of the quite high level (for A-Level) maths this was somewhat inevitable. Also Python probably shouldn't have been used (abused) to make this.


## Repository structure

- `src/` — application source code
- `tests/scripts/` — test and utility scripts
- `tests/fixtures/` — organised test inputs and generated test fixtures
- `data/samples/` — canonical source audio and generated sample files
- `report/source/` — LaTeX report source and report assets
- `report/report.pdf` — compiled report
- `docs/notes/` — project notes and diagrams
