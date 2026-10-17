# NeuroScreen: EEG depression screening dashboard

Everything is in this one folder: plain HTML, CSS and JavaScript for the dashboard, Python for the server and the machine-learning model, and the dataset.

```
index.html  style.css  script.js     the web dashboard
app.py                               Python server (FastAPI): serves the page and the API
common.py                            EEG filtering, epoching and feature extraction
train.py                             trains and evaluates the model (subject-wise splits)
gen_data.py                          makes the synthetic demo dataset
dataset/                             40 training recordings + labels.csv
models/                              trained model + its metrics
sample_recordings/                   files to upload in the dashboard (not part of the training set)
requirements.txt  run_windows.bat  run_mac_linux.sh
```

**Demo mode.** The bundled dataset is SYNTHETIC, so the page shows a "Demo" badge and banner. Results on it say nothing about real patients. It is a research prototype, not a clinical diagnostic tool.

## Run

Windows: double-click `run_windows.bat`. Mac/Linux: `./run_mac_linux.sh`.

Or by hand (Python 3.10+):

```bash
pip install -r requirements.txt
python app.py
```

The browser opens at http://localhost:8000. Use another port with `python app.py --port 8080`.

Try it: click "Load healthy sample" or "Load MDD sample", or upload any file from `sample_recordings/`.

The first start uses the saved model in `models/`. If it was made by a different scikit-learn version, the app retrains it automatically from `dataset/` (about a minute).

## Use your real dataset

1. Put your recordings in `dataset/` and write `dataset/labels.csv` with columns `file,subject,label` (1 = MDD, 0 = healthy).
2. Delete `dataset/SYNTHETIC`. The Demo badge, banner and sample buttons then disappear.
3. Edit `load_recording()` in `common.py` so it returns a `(n_channels, n_samples)` array in microvolts at 250 Hz. For .edf or .set files use MNE: `mne.io.read_raw_edf(p, preload=True).pick(CHANNELS).resample(250).get_data() * 1e6`.
4. Match `CHANNELS` and `FS` in `common.py` to your dataset.
5. Run `python train.py`, then restart the app.

For a 9 GB dataset, train on your own machine or Google Colab. `train.py` reads one file at a time, so memory use stays low.

## Upload format

`.npy` with shape (19 channels, samples) or `.csv` with one row per sample and 19 columns. 250 Hz, at least 8 seconds, 10-20 montage in the channel order listed in `dataset/README.txt`.

## Notes

- Splits are always by subject, never by epoch. Splitting by epoch leaks data and gives falsely high accuracy.
- The model is evaluated with 5-fold subject-wise cross-validation repeated 5 times over all subjects. A single small hold-out set (about 10 people) swings too much to trust.
- The bundled dataset is small on purpose (45 s per recording, float16) so the whole project fits in one zip. `DUR=90 python gen_data.py` makes a larger one.
- Screening output reflects EEG patterns only. It cannot replace a clinical interview.
