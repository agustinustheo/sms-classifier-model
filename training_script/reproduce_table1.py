"""Rebuild Table I of the EIConCIT 2021 paper in one command (run from training_script/).

    uv run reproduce_table1.py            # 8 models x 10 folds x 2 batches + external test
    uv run reproduce_table1.py --quick    # 3-fold smoke test with 3 fast models

Table I has four columns per model:
  - 10-fold CV accuracy on batch 1 (data.txt, 4,125 SMS) and batch 2 (data_1;1;1.txt, 1,260 SMS)
  - accuracy on the external Rahmi & Wibisono (2016) dataset, training on batch 1 / batch 2

This re-implements the 2020 pipeline (stratified_k_fold_per_ratio.py) with its quirks, which the
numbers depend on: training and test messages are preprocessed differently, and the 1,500-word
vocabulary in word_features.pickle is reused as-is. Unlike 2020, every model and shuffle is seeded,
so runs are deterministic. It is also much faster, because features are computed once per message.

It also reports duplicate-aware CV (not in the paper). The full dataset has only ~1,900 distinct
texts among 4,125 rows, so ordinary k-fold tests models on exact copies of training messages.

Outputs go to results/: table1_reproduced.md, table1_kfold_batch{1,2}.csv, table1_run_info.json.
"""

import argparse
import csv
import io
import json
import pickle
import pickletools
import platform
import re
import time
from importlib.metadata import version
from pathlib import Path

import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

HERE = Path(__file__).resolve().parent
DATA = {"batch1": HERE / "corpus/sms_corpus/data.txt",
        "batch2": HERE / "corpus/sms_corpus/data_1;1;1.txt"}
RAHMI = HERE / "corpus/external/rahmi2016/dataset_sms_spam_v1.csv"
VOCAB = HERE / "word_features.pickle"
OUT = HERE / "results"
SEED = 1

# Paper, Table I (accuracy %).
PAPER = {
    "MNB": (92.68, 92.46, 57.65, 57.56), "MLR": (94.57, 94.60, 78.39, 83.90),
    "SVM": (94.38, 93.89, 77.16, 82.62), "KNN": (86.21, 74.76, 60.54, 62.37),
    "DT": (93.65, 90.24, 72.26, 65.79), "SGD": (94.50, 93.49, 75.50, 82.85),
    "XGB": (94.52, 93.50, 77.34, 83.20), "RF": (94.62, 95.00, 77.25, 82.67),
}
COLUMNS = ["cv_batch1", "cv_batch2", "external_batch1", "external_batch2"]

# Labels: this dataset uses 0 ham, 1 promo, 2 spam; Rahmi uses 0 normal, 1 fraud (spam), 2 promo.
RAHMI_TO_OURS = {0: 0, 1: 2, 2: 1}

PHONE = r"(\()?(\+62|62|0)(\d{2,3})?\)?[ .-]?\d{2,4}[ .-]?\d{2,4}[ .-]?\d{2,4}"
URL = r"[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)"
EMAIL = r"^.+@[^\.].*\.[a-z]{2,}$"


# ---------------------------------------------------------------- data

def load_corpus(path):
    texts, labels = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if "<%>" in line:
            label, text = line.split("<%>", 1)
            labels.append(int(label))
            texts.append(text)
    return texts, np.array(labels)


def load_rahmi(path):
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [r["Teks"] for r in rows], np.array([RAHMI_TO_OURS[int(r["label"])] for r in rows])


def load_vocab(path):
    """word_features.pickle is a plain list of strings; refuse anything that could run code."""
    data = path.read_bytes()
    unsafe = {"GLOBAL", "STACK_GLOBAL", "REDUCE", "INST", "OBJ", "NEWOBJ", "NEWOBJ_EX", "BUILD"}
    if {op.name for op, _, _ in pickletools.genops(data)} & unsafe:
        raise ValueError(f"{path} is not a plain list pickle")
    return pickle.loads(data)


# ---------------------------------------------------------------- 2020 preprocessing

def deleet(text):
    """convert_tacky_text: digits to letters inside mixed tokens ("m4ma" -> "mama")."""
    table = str.maketrans("1234567890", "iseasgtbgo")
    out = []
    for tok in word_tokenize(text):
        if any(c.isalpha() for c in tok) and any(c.isdigit() for c in tok):
            tok = tok.lower()
            if not (tok.endswith("ribu") or tok.endswith("rbu") or tok.endswith("rb")):
                tok = tok.translate(table)
        out.append(tok)
    return " ".join(out) + " "


def preprocess_train(text):
    """preproccess_df (pandas str.replace = regex). Its de-leet step was a no-op in 2020."""
    p = text.lower()
    p = re.sub(EMAIL, " emailaddress ", p)
    p = re.sub(PHONE, " phonenumber", p)
    p = re.sub(URL, " webaddress ", p)
    p = p.replace("http", "").replace("https", "")
    p = re.sub(r"£|\$", " moneysymbol ", p)
    p = re.sub(r" rp.", " moneysymbol ", p)
    p = re.sub(r" rp", " moneysymbol ", p)
    p = re.sub(r"\d+(\.\d+)?", " number ", p)
    p = re.sub(r"[.,\/#!%\^&\*;:{}=\-_`~()?]", " ", p)
    p = re.sub(r"\s+", " ", p)
    p = re.sub(r"[^a-zA-Z]", " ", p)
    return re.sub(r"^\s+|\s+?$", "", p)


def preprocess_test(text):
    """preproccess_text, used for held-out and external messages."""
    p = deleet(text.lower())
    p = re.sub(EMAIL, " emailaddress ", p)
    p = re.sub(PHONE, " phonenumber ", p)
    p = re.sub(URL, " webaddress ", p)
    p = p.replace("http", "").replace("https", "")
    p = re.sub(r"£|\$", "moneysymbol ", p)
    p = p.replace(" rp.", " moneysymbol ").replace(" rp", " moneysymbol ")
    p = re.sub(r"\d+(\.\d+)?", " number ", p)
    p = re.sub(r"[.,\/#!%\^&\*;:+{}=\-_`~()?]", " ", p)
    p = re.sub(r"\s+", " ", p)
    return re.sub(r"^\s+|\s+?$", "", p)


def featurize(texts, vocab):
    """find_features: binary presence of each vocabulary word."""
    index = {w: i for i, w in enumerate(vocab)}
    X = np.zeros((len(texts), len(vocab)))
    for r, t in enumerate(texts):
        for tok in set(word_tokenize(t)):
            if tok in index:
                X[r, index[tok]] = 1.0
    return X


# ---------------------------------------------------------------- models

def make_models():
    """Same estimators and (default) hyperparameters as 2020, plus fixed seeds."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from xgboost import XGBClassifier

    return {
        "MNB": MultinomialNB(),
        "MLR": LogisticRegression(),
        "SVM": SVC(kernel="linear"),
        "KNN": KNeighborsClassifier(),
        "DT": DecisionTreeClassifier(random_state=SEED),
        "SGD": SGDClassifier(max_iter=100, random_state=SEED),
        "XGB": XGBClassifier(random_state=SEED),
        "RF": RandomForestClassifier(random_state=SEED),
    }


def fit(model, X, y, seed):
    order = np.random.default_rng(seed).permutation(len(y))  # 2020 shuffled before fitting
    return clone(model).fit(X[order], y[order])


def cross_validate(model, Xtr, Xte, y, splits):
    return [100 * accuracy_score(y[te], fit(model, Xtr[tr], y[tr], SEED + k).predict(Xte[te]))
            for k, (tr, te) in enumerate(splits, 1)]


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="3 folds, MNB/MLR/DT only")
    args = ap.parse_args()
    folds = 3 if args.quick else 10

    nltk.download("punkt_tab", quiet=True)
    OUT.mkdir(exist_ok=True)
    vocab = load_vocab(VOCAB)
    models = make_models()
    if args.quick:
        models = {k: v for k, v in models.items() if k in ("MNB", "MLR", "DT")}

    rahmi_texts, y_rahmi = load_rahmi(RAHMI)
    X_rahmi = featurize([preprocess_test(t) for t in rahmi_texts], vocab)

    results = {m: {} for m in models}
    t0 = time.time()
    for batch, path in DATA.items():
        texts, y = load_corpus(path)
        print(f"\n{batch}: {len(y)} messages, class counts {np.bincount(y).tolist()}", flush=True)
        Xtr = featurize([preprocess_train(t) for t in texts], vocab)
        Xte = featurize([preprocess_test(t) for t in texts], vocab)
        splits = list(StratifiedKFold(folds, shuffle=True, random_state=SEED).split(Xtr, y))
        groups = np.unique([" ".join(t.lower().split()) for t in texts], return_inverse=True)[1]
        gsplits = list(StratifiedGroupKFold(folds, shuffle=True, random_state=SEED).split(Xtr, y, groups))

        rows = []
        for name, model in models.items():
            ts = time.time()
            accs = cross_validate(model, Xtr, Xte, y, splits)
            gaccs = cross_validate(model, Xtr, Xte, y, gsplits)
            ext = 100 * accuracy_score(y_rahmi, fit(model, Xtr, y, SEED).predict(X_rahmi))
            results[name].update({f"cv_{batch}": np.mean(accs), f"dedup_cv_{batch}": np.mean(gaccs),
                                  f"external_{batch}": ext})
            rows.append({"model": name, **{f"fold {i}": f"{a:.2f}" for i, a in enumerate(accs, 1)}})
            print(f"  {name:4s} cv {np.mean(accs):6.2f}  dedup-cv {np.mean(gaccs):6.2f}  "
                  f"external {ext:6.2f}  ({time.time() - ts:.0f}s)", flush=True)
        with open(OUT / f"table1_kfold_{batch}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["model", *[f"fold {i}" for i in range(1, folds + 1)]])
            w.writeheader()
            w.writerows(rows)

    lines = [
        "# Table I: paper vs. reproduced",
        "",
        f"Generated by `reproduce_table1.py` ({folds}-fold, anonymised corpus). Accuracy in %; each cell is paper / reproduced (difference).",
        "",
        "| Model | CV batch 1 | CV batch 2 | External, trained on batch 1 | External, trained on batch 2 |",
        "|---|---|---|---|---|",
    ]
    for m, r in results.items():
        cells = [f"{PAPER[m][i]:.2f} / {r[c]:.2f} ({r[c] - PAPER[m][i]:+.2f})" for i, c in enumerate(COLUMNS)]
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines += ["", "## Duplicate-aware cross-validation (not in the paper)", "",
              "Identical messages are kept in the same fold, so models are never tested on an exact copy of a training message.", "",
              "| Model | CV batch 1 | dedup CV batch 1 | CV batch 2 | dedup CV batch 2 |", "|---|---:|---:|---:|---:|"]
    for m, r in results.items():
        lines.append(f"| {m} | {r['cv_batch1']:.2f} | {r['dedup_cv_batch1']:.2f} | {r['cv_batch2']:.2f} | {r['dedup_cv_batch2']:.2f} |")
    (OUT / "table1_reproduced.md").write_text("\n".join(lines) + "\n")

    info = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - t0), "folds": folds,
            "python": platform.python_version(), "platform": platform.platform(),
            "packages": {p: version(p) for p in ["scikit-learn", "xgboost", "nltk", "numpy"]}}
    (OUT / "table1_run_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"\nDone in {info['seconds']}s. See results/table1_reproduced.md")


if __name__ == "__main__":
    main()
