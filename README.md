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
- **[`original`](../../tree/original)**: the code exactly as it was for the paper (commit `40eb95d`, September 2020),
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

## Reproducing Table I

```bash
cd training_script && uv run reproduce_table1.py      # ~10 min on a modern laptop
```

This rebuilds all of Table I: 10-fold CV on both training batches, plus the external test on Rahmi & Wibisono's dataset.
It writes `results/table1_reproduced.md` with the paper's numbers alongside. It re-implements the 2020 pipeline,
including its quirks, but seeds every model, so runs are deterministic.

Latest run (anonymised corpus, full details in [`training_script/results/table1_reproduced.md`](training_script/results/table1_reproduced.md)):

- 24 of the 32 cells are within 2 points of the paper and 30 within 3, and the conclusions hold. RF and the linear models lead
  (93.5–94.8% CV), KNN is worst and drops sharply on the balanced batch, and MNB scores ~58% on the external set.
- Most deviations are in the high-variance tree model. The largest is DT tested externally after training on batch 2:
  75.0% vs the paper's 65.8%. The 2020 runs weren't seeded, so single runs of DT/KNN vary.
- Duplicate-aware CV is 1.5–6 points lower than the paper-style CV (e.g. RF 94.2 → 91.9 on batch 1).

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

The paper's external test set is in `training_script/corpus/external/rahmi2016/`: Rahmi & Wibisono (2016), 1,143 SMS
(569 normal, 335 fraud, 239 promo), unmodified from
[Yudi Wibisono's release](https://yudiwbs.wordpress.com/2018/08/05/dataset-klasifikasi-bahasa-indonesia-sms-spam-klasifikasi-teks-dengan-scikit-learn/)
under CC BY-SA 4.0. Its labels are `0` normal, `1` fraud, `2` promo, so `reproduce_table1.py` maps them to this dataset's scheme.
Cite it as: Rahmi, F. and Wibisono, Y. (2016). *Aplikasi SMS Spam Filtering pada Android menggunakan Naive Bayes*. Unpublished manuscript.

Known issues with the data, worth knowing before comparing models:

- **Duplicates.** `data.txt` has 4,125 rows but only about 1,900 distinct messages (operator messages repeat up to 42 times),
  and 86 texts carry conflicting labels. Ordinary k-fold therefore tests on copies of training messages, which inflates
  accuracy by 2 to 6 points (see the duplicate-aware CV above).
- `test_data.txt` (201 rows) is not the external test set, and 197 of its rows also appear in `data.txt`.

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


## License

This repository is licensed under [CC BY-SA 4.0](LICENSE). If you use the code, data or results, please cite the paper above.
The external dataset in `training_script/corpus/external/rahmi2016/` is © Rahmi & Wibisono, also CC BY-SA 4.0.

## Repository history

The whole history (including `original`) was rewritten in November 2026:

- the SMS data is anonymised in every commit (see `anonymise_corpus.py`);
- a third-party Indonesian news corpus (Kompas/Tempo articles), once used to train a sentence tokenizer, was removed;
- the licence was added from the first commit on.

Commit hashes from before November 2026 no longer exist.
