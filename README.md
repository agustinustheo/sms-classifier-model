# Indonesian SMS Classifier

Classifies Indonesian SMS as **ham**, **promo** or **spam** with eight classic ML models
(Multinomial NB, Logistic Regression, linear SVM, KNN, Decision Tree, SGD, XGBoost, Random Forest)
on a 1,500-word bag-of-words. It uses NLTK for text processing and scikit-learn / XGBoost for classification.

This is the code behind:

> A. Theodorus, T. K. Prasetyo, R. Hartono, D. Suhartono. **Short Message Service (SMS) Spam Filtering using
> Machine Learning in Bahasa Indonesia.** 2021 3rd East Indonesia Conference on Computer and Information
> Technology (EIConCIT), pp. 199–202. doi:[10.1109/EIConCIT50028.2021.9431859](https://doi.org/10.1109/EIConCIT50028.2021.9431859)

## Branches

- **`master`**: modernised for current Python (3.11+) and libraries, managed with [uv](https://docs.astral.sh/uv/).
- **[`original`](../../tree/original)**: the code exactly as it was for the paper (commit `dda2997`, September 2020),
  including the trained models from that time.

## Quick start

```bash
# install uv: https://docs.astral.sh/uv/getting-started/installation/
uv sync

# train the models on the full dataset (writes sms_classifier_pickle/ratio(5.5;3.3;1)/)
cd training_script && uv run training_script.py && cd ..

# classify with the trained models
uv run run_sms_classifier_model.py
```

Other scripts (run them from `training_script/`):

| Script | What it does |
|---|---|
| `training_script_per_ratio.py` | trains models for every class ratio (`1;1;1`, `1;1;2`, `1;2;1`, `2;1;1`, full) |
| `stratified_k_fold_per_ratio.py` | stratified 10-fold CV per model and ratio → `results/*.csv` (the paper's Table I CV columns and Figs. 2–3) |
| `dataset_average_length.py` | average message length per class |

`run_sms_classifier_model_per_ratio.py` (repo root) needs the per-ratio models trained first.

## Data

`training_script/corpus/sms_corpus/`, in `label<%>text` format (`0` ham, `1` promo, `2` spam):

- `data.txt`: full dataset, 4,125 messages (2,323 ham / 1,382 promo / 420 spam). Used as batch 1 in the paper.
- `data_1;1;1.txt`: balanced 420/420/420 sample. Used as batch 2.
- `data_1;1;2.txt`, `data_1;2;1.txt`, `data_2;1;1.txt`: other ratios.

The messages come from the authors' own phones and are **anonymised**. Phone numbers, e-mail addresses, names in shared
contacts and digit runs of 7 or more were replaced with `phonenumber`, `emailaddress`, `[Name]nama` and `number`
(`training_script/anonymise_corpus.py`). Names inside ordinary messages were not masked.
Masking barely changes the results: balanced-batch 10-fold CV gives SVM 93.6%, SGD 93.7% and MNB 92.1%,
against 93.89%, 93.49% and 92.46% in the paper.

The paper's external test set is Rahmi & Wibisono (2016), 1,143 SMS, CC BY-SA 4.0. It's available from
[Yudi Wibisono's blog](https://yudiwbs.wordpress.com/2018/08/05/dataset-klasifikasi-bahasa-indonesia-sms-spam-klasifikasi-teks-dengan-scikit-learn/).
Note that its labels are `0` normal, `1` fraud, `2` promo.

## What changed from `original`

- `requirements.txt` replaced by `pyproject.toml` + `uv.lock` (Python 3.12, pandas 3, scikit-learn 1.9, XGBoost 3, NLTK 3.10).
- `str.replace(...)` calls now pass `regex=True`. pandas 2 changed the default to literal matching, which silently
  broke the preprocessing. With the flag, the output is identical to 2020 for every message in `data.txt`.
- NLTK's tokenizer data (`punkt_tab`) is downloaded automatically on first run.
- `np.random.seed = seed` (which replaced the function instead of seeding) is now `np.random.seed(seed)`,
  so the data shuffle is reproducible.
- The 2020 model pickles were removed. Half of them (DT, RF, SVM, XGBoost) can't be loaded by current
  scikit-learn / XGBoost, and loading pickles from the internet is a security risk anyway. Re-train them with
  the scripts above, or use the `original` branch with a 2020 environment.
- The training scripts create their output folders, and a broken import in `dataset_average_length.py` was fixed.

The rest is unchanged, including the known quirks that the paper's numbers depend on: training and test messages are
preprocessed slightly differently, and the vocabulary in `word_features.pickle` is the first 1,500 words seen
rather than the most frequent ones.

Not modernised: `training_script/tensorflow_script/` (an exploratory Keras notebook, not used in the paper).

### Troubleshooting

NLTK ≥ 3.9.3 refuses data directories that other users can write to. If `word_tokenize` reports that
`~/nltk_data` is "writable by other accounts", run `chmod -R go-w ~/nltk_data`.

## Credits

We couldn't have assembled this model without the help of some awesome creators on the internet:
Harrison Kinsley a.k.a. sentdex, Laurence Moroney and Eduonix.

- Sentdex, [NLTK with Python 3 for Natural Language Processing](https://www.youtube.com/watch?v=FLZvOKSCkxY&list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL)
- Eduonix, [NLP for Text Classification with NLTK & Scikit-learn](https://www.youtube.com/watch?v=G4UVJoGFAv0)
- Laurence Moroney, [Text Classification Tutorial (Coding TensorFlow)](https://www.youtube.com/watch?v=BO4g2DRvL6U)
