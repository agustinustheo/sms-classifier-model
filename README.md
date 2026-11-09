# Indonesian SMS Classifier

This repository contains code for *Short Message Service (SMS) Spam Filtering using Machine Learning in Bahasa Indonesia*
([EIConCIT 2021](https://doi.org/10.1109/EIConCIT50028.2021.9431859)).

It classifies Indonesian text messages as **ham**, **promo** or **spam**. Messages are cleaned and turned into a
1,500-word bag of words with NLTK, then classified with eight models: Multinomial Naive Bayes, Logistic Regression,
linear SVM, k-Nearest Neighbours, Decision Tree, SGD, XGBoost and Random Forest. Random Forest and Logistic Regression
perform best, at around 94.5% accuracy in 10-fold cross-validation.

The code as it was for the paper is on the [`original`](../../tree/original) branch.

## Installation

Clone the repository and install the dependencies with [uv](https://docs.astral.sh/uv/):

```
git clone https://github.com/agustinustheo/sms-classifier-model.git
cd sms-classifier-model
uv sync
```

## Training and classifying

Train all eight models on the full dataset, then classify the test messages with them:

```
cd training_script && uv run training_script.py && cd ..
uv run run_sms_classifier_model.py
```

`training_script/training_script_per_ratio.py` trains the models for each class ratio (`1;1;1`, `1;1;2`, `1;2;1`,
`2;1;1` and the full set), and `run_sms_classifier_model_per_ratio.py` runs them.

## Reproducing the paper's results

```
cd training_script
uv run reproduce_table1.py
```

This rebuilds Table I of the paper: 10-fold cross-validation on the full and balanced datasets, and the test on the
external dataset. It writes the results next to the paper's numbers in
[`training_script/results/table1_reproduced.md`](training_script/results/table1_reproduced.md).

The same messages often appear more than once in the dataset, so the script also reports cross-validation with
duplicates kept in the same fold. Those scores are a few points lower and are a fairer estimate for unseen messages.

## Dataset

The messages are in `training_script/corpus/sms_corpus/`, one per line as `label<%>text` (`0` ham, `1` promo, `2` spam):

- `data.txt`: 4,125 messages (2,323 ham, 1,382 promo, 420 spam), collected from the authors' phones since October 2018
- `data_1;1;1.txt`: a balanced sample of 420 messages per class
- `data_1;1;2.txt`, `data_1;2;1.txt`, `data_2;1;1.txt`: samples with other class ratios

Phone numbers, e-mail addresses and long account numbers are masked (`training_script/anonymise_corpus.py`).

The external test set in `training_script/corpus/external/rahmi2016/` is the Indonesian SMS spam dataset by
Ferdila Rahmi and Yudi Wibisono: 1,143 messages (569 normal, 335 fraud, 239 promo) from Rahmi's research at
Universitas Pendidikan Indonesia ([thesis](https://repository.upi.edu/26936/)), published by Yudi Wibisono
([source](https://yudiwbs.wordpress.com/2018/08/05/dataset-klasifikasi-bahasa-indonesia-sms-spam-klasifikasi-teks-dengan-scikit-learn/))
under CC BY-SA 4.0. Thank you to both for sharing it. If you use it, please cite it too (see [Citing](#citing)).

## Highlevel overview of source files

- `run_sms_classifier_model.py`, `run_sms_classifier_model_per_ratio.py`: classify messages with the trained models.
- `training_script/training_script.py`, `training_script_per_ratio.py`: preprocessing, feature extraction and training.
- `training_script/stratified_k_fold_per_ratio.py`: 10-fold cross-validation per model and class ratio.
- `training_script/reproduce_table1.py`: rebuilds Table I of the paper.
- `training_script/word_features.pickle`: the 1,500-word vocabulary used as features.
- `training_script/tensorflow_script/`: an early experiment with a Keras text classifier.

## Acknowledgements

We couldn't have assembled this model without the help of some awesome creators on the internet:

- Harrison Kinsley (sentdex), [NLTK with Python 3 for Natural Language Processing](https://www.youtube.com/watch?v=FLZvOKSCkxY&list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL)
- Eduonix, [NLP for Text Classification with NLTK & Scikit-learn](https://www.youtube.com/watch?v=G4UVJoGFAv0)
- Laurence Moroney, [Text Classification Tutorial (Coding TensorFlow)](https://www.youtube.com/watch?v=BO4g2DRvL6U)

## License

[CC BY-SA 4.0](LICENSE), the same licence as the external dataset.

## Citing

If you find this repository useful in your research, we'd really appreciate it if you cite our paper:

    @inproceedings{theodorus2021sms,
      title={Short Message Service (SMS) Spam Filtering using Machine Learning in Bahasa Indonesia},
      author={Theodorus, Agustinus and Prasetyo, Tio Kristian and Hartono, Reynaldi and Suhartono, Derwin},
      booktitle={2021 3rd East Indonesia Conference on Computer and Information Technology (EIConCIT)},
      year={2021},
      pages={199--202},
      organization={IEEE},
      doi={10.1109/EIConCIT50028.2021.9431859}
    }

If you use the external test set, please also cite Rahmi and Wibisono's work, as they ask:

> Rahmi, F. and Wibisono, Y. (2016). Aplikasi SMS Spam Filtering pada Android menggunakan Naive Bayes, Unpublished manuscript.

    @unpublished{rahmi2016sms,
      title={Aplikasi SMS Spam Filtering pada Android menggunakan Naive Bayes},
      author={Rahmi, Ferdila and Wibisono, Yudi},
      year={2016},
      note={Unpublished manuscript}
    }

GitHub's "Cite this repository" button gives our paper in other formats.
