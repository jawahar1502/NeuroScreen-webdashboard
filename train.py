"""Train and evaluate with SUBJECT-WISE splits (no epoch leakage).

Evaluation: repeated stratified group k-fold cross-validation over ALL subjects.
Every subject is always scored by a model that never saw that person. Scoring every
subject this way (instead of one small hold-out set) gives a far steadier estimate,
because a single split of ~10 people swings wildly depending on who lands in it.
The deployed model is then trained on all subjects.
"""
import json, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score
from common import load_recording, recording_to_features, feature_names, BANDS, CHANNELS

N_FOLDS, N_REPEATS = 5, 5
BASE = Path(__file__).resolve().parent
DATA = BASE / "dataset"
lab = pd.read_csv(DATA / "labels.csv")
X, y, g = [], [], []
for _, r in lab.iterrows():
    f = recording_to_features(load_recording(DATA / r.file))
    X.append(f); y += [r.label] * len(f); g += [r.subject] * len(f)
X, y, g = np.vstack(X), np.array(y), np.array(g)
print("epochs:", X.shape, "subjects:", lab.subject.nunique())


def model():
    return make_pipeline(StandardScaler(),
                         RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                                n_jobs=-1, random_state=0))


def subject_scores(m, Xs, ys, gs):
    """Average epoch probabilities per subject -> one prediction per recording."""
    p = m.predict_proba(Xs)[:, 1]
    return pd.DataFrame({"g": gs, "p": p, "y": ys}).groupby("g").mean()


# repeated subject-wise cross-validation over all subjects
fold_acc = []
oof = {s: [] for s in lab.subject}
for rep in range(N_REPEATS):
    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=rep)
    for a, b in skf.split(X, y, g):
        m = model().fit(X[a], y[a])
        df = subject_scores(m, X[b], y[b], g[b])
        fold_acc.append(float(((df.p > 0.5) == (df.y > 0.5)).mean()))
        for s, p in df.p.items():
            oof[s].append(float(p))

y_sub = lab.label.values.astype(int)
p_sub = np.array([np.mean(oof[s]) for s in lab.subject])   # each subject's out-of-fold probability
pred = (p_sub > 0.5).astype(int)
tn, fp, fn, tp = confusion_matrix(y_sub, pred, labels=[0, 1]).ravel()
metrics = {
    "cv_accuracy_mean": float(np.mean(fold_acc)), "cv_accuracy_std": float(np.std(fold_acc)),
    "accuracy": float((pred == y_sub).mean()),
    "auc": float(roc_auc_score(y_sub, p_sub)), "f1": float(f1_score(y_sub, pred)),
    "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    "n_subjects": int(len(lab)), "cv_folds": N_FOLDS, "cv_repeats": N_REPEATS,
    "demo": (DATA / "SYNTHETIC").exists(),
    "data_note": "SYNTHETIC demo data. Replace with real EEG and retrain.",
}

# final model on all subjects for deployment
final = model().fit(X, y)
rf = final[-1]
imp = sorted(zip(feature_names(), rf.feature_importances_), key=lambda t: -t[1])[:10]
metrics["top_features"] = [{"name": n, "importance": float(v)} for n, v in imp]

# class reference profile: mean relative power per band (averaged over channels)
nb, nc = len(BANDS), len(CHANNELS)
rel = X[:, :nb * nc].reshape(len(X), nb, nc).mean(2)
metrics["reference"] = {
    "bands": list(BANDS),
    "healthy": rel[y == 0].mean(0).tolist(), "mdd": rel[y == 1].mean(0).tolist(),
}
(BASE / "models").mkdir(exist_ok=True)
joblib.dump(final, BASE / "models" / "model.joblib")
json.dump(metrics, open(BASE / "models" / "metrics.json", "w"), indent=2)
print(json.dumps({k: v for k, v in metrics.items() if k not in ("reference", "top_features")}, indent=2))
