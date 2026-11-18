# Intelligent Toxicity Moderation Platform

CSE303 Machine Learning — Capstone Project (D. Abhiram, AP24110010593).
Repository: [abhi-2667/cse303-ml-ap24110010593-toxicity-moderation](https://github.com/abhi-2667/cse303-ml-ap24110010593-toxicity-moderation)

Multi-label toxic-comment detection on the full
[Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
data (159,571 comments), a 3-tier moderation engine (`PUBLISH` / `REVIEW` / `REMOVE`) and a Streamlit demo.

## The problem
Predict the probability of 6 kinds of toxicity at once: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`.

**Errors are not equally costly.** A false positive is a clean comment removed (unfair censorship); a false negative is hate speech or a threat left online. So instead of one yes/no cut-off the system uses a human `REVIEW` lane and a lower bar for `threat` / `severe_toxic`.

**Twist:** word TF-IDF (meaning) + character TF-IDF (spelling), aimed at disguised words like `1d10t` / `i.d.i.o.t`. Measured result: it improves overall ROC-AUC (0.9751 -> 0.9846), but its effect on disguised insults specifically is weak (see `REPORT.md`).

## Two ways to read the project
| What | Where |
|------|-------|
| **Complete single notebook** (instructor requirement: objective -> data -> EDA -> features -> 3 models -> robustness test -> decision engine) | `notebooks/toxicity_moderation_capstone.ipynb` |
| **Lab-schedule series** (one notebook per stage of the course ladder) | `notebooks/01_eda` ... `06_clustering` |

| Notebook | Lab link | Content |
|----------|----------|---------|
| `01_eda` | — | imbalance, co-occurrence, length, top words, **how common are disguised words** |
| `02_preprocessing` | Exp 7, 8 | cleaning, leakage-free split, hybrid TF-IDF, chi-square, SVD (PCA idea) |
| `03_baseline_model` | Exp 3–6 | Logistic Regression, KNN, Decision Tree |
| `04_model_comparison` | Exp 9, 10, Unit V | SVM, Naive Bayes, Random Forest + comparison table |
| `05_tuning_and_benchmark` | — | tune top 2, 3 random splits (mean ± std), benchmark gap, business decision numbers |
| `06_clustering` | Exp 14, 15 | K-Means (elbow + silhouette) and Hierarchical clustering of toxic comments |

The neural-network stage (Exp 11–13) is intentionally not part of this project.

## Project structure
```
data/raw/          dataset (git-ignored)           data/processed/  saved features (git-ignored)
models/            deployed .pkl (force-added)     assets/          plots used in the report
notebooks/         the 7 notebooks                 app/app.py       Streamlit demo (mandatory deployment)
download_data.py   dataset download + fallback     PROPOSAL.md, REPORT.md, VIVA_NOTES.md
```

## How to run
```bash
python -m venv venv && source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

python download_data.py          # real dataset (Kaggle API -> GitHub mirror -> tiny synthetic fallback)

# run notebooks in this order (open in Jupyter and "Run All", or from the terminal):
cd notebooks
for nb in 01_eda 02_preprocessing 03_baseline_model 04_model_comparison 05_tuning_and_benchmark 06_clustering toxicity_moderation_capstone; do
  jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 $nb.ipynb
done
cd ..
streamlit run app/app.py
```
Notebooks 03–06 need the files saved by notebook 02. On a laptop the full run takes roughly an hour and needs about 4 GB of free RAM (character TF-IDF and the SVM are the heavy parts). For a quick test, set `FRACTION = 0.05` in notebook 02 and `SAMPLE_FRACTION = 0.05` in the capstone notebook.

### Kaggle setup (optional)
Put `kaggle.json` in `~/.kaggle/` (Windows: `C:\Users\<you>\.kaggle\`) and accept the competition rules once on the competition page. If Kaggle is not set up, `download_data.py` downloads the same `train.csv` (CC0 dataset) from a public GitHub mirror and checks it has exactly 159,571 rows.

## Live demo (Streamlit Community Cloud)
[![Deploy to Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=abhi-2667/cse303-ml-ap24110010593-toxicity-moderation&branch=main&mainModule=app/app.py)

The app is deploy-ready: `models/best_model.pkl` and `models/tfidf_union.pkl` (~1 MB each) are committed on purpose so the Cloud container does not need to retrain. To publish it: click the badge, sign in with GitHub, keep the repo / branch `main` / entrypoint `app/app.py`, pick Python 3.11, and press **Deploy**. Replace the badge with the `https://<app-name>.streamlit.app` link once it is live.

Local run: `streamlit run app/app.py` (tested: page returns HTTP 200 and the badge/plot work).

## Moderation rules
| Decision | Rule |
|----------|------|
| PUBLISH | max probability < 0.40 |
| REVIEW  | 0.40 <= max probability <= 0.75 (human moderator) |
| REMOVE  | max probability > 0.75, or `threat` / `severe_toxic` > 0.60 |

## Git tips (from the course guide)
```bash
git tag milestone     # after the SVM / comparison stage
git tag final         # final state before Poster Day
```
Both tags already exist here: `milestone` sits on the model-comparison commit, `final` on the last commit.

## Sources and citations
* **Kaggle competition page (dataset + metric):** [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) — Jigsaw / Conversation AI, 2018. `train.csv` is released under CC0; the comment text is from Wikipedia (CC-BY-SA 3.0). Metric: mean column-wise ROC-AUC.
* **GitHub mirror used by `download_data.py`:** [iampukar/toxic-comments-classification](https://github.com/iampukar/toxic-comments-classification) — public copy of `train.csv` (checked: 159,571 rows, official label counts).
* **Baseline reference:** NYC Data Science Academy, ["Jigsaw's Text Classification Challenge"](https://nycdatascience.com/blog/student-works/jigsaws-text-classification-challenge/) (2018) — TF-IDF + Logistic Regression at about 0.976 public leaderboard ROC-AUC.

## Notes
* `app/app.py` loads `models/tfidf_union.pkl` and `models/best_model.pkl`, which are created by the capstone notebook (trained on the 80% training split, so the validation scores describe exactly the deployed model). Keep `clean_text()` in the app identical to the notebook's version.
* Dataset: Kaggle Toxic Comment Classification Challenge (Jigsaw / Conversation AI), CC0; comment text is from Wikipedia (CC-BY-SA 3.0).
