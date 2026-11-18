# REPORT — Intelligent Toxicity Moderation Platform
CSE303 Capstone | D. Abhiram, AP24110010593

All numbers below come from the executed notebooks on the **full Jigsaw `train.csv` (159,571 comments)**, 80/20 stratified split, fixed seeds. Validation scores are on a split of `train.csv`, not on Kaggle's hidden test set.

## 1. Problem and business framing
Predict six toxicity labels per comment (multi-label) and turn the probabilities into one of three actions: **PUBLISH** (max prob < 0.40), **REVIEW** (0.40–0.75, human moderator), **REMOVE** (> 0.75, or `threat` / `severe_toxic` > 0.60). A false positive removes a normal user's comment (censorship); a false negative leaves hate speech or threats online, so the costs are asymmetric and a human REVIEW lane exists.

## 2. Data and EDA (`01_eda.ipynb`)
* No missing values; **89.8%** of comments carry no label; `threat` (478 comments) and `identity_hate` (1,405) are very rare -> accuracy is useless, so ROC-AUC (column-averaged) and Macro F1 are used.
* Labels overlap strongly (`toxic`, `obscene`, `insult`) -> multi-label setup.
* **Disguised-word check:** simple regex patterns for `f*ck`-style, `l0ser`-style and `i.d.i.o.t`-style spellings match about 1% of comments and are *not* clearly more common in toxic comments (e.g. letter-symbol-letter: 1.48% of toxic vs 1.73% of clean). Only dotted letters differ (0.25% vs 0.09%). The regexes are crude, so this is an indication only.

## 3. Method
Clean text (lowercase, strip HTML/URLs, normalise spaces; digits and punctuation kept) -> **split first** -> hybrid `FeatureUnion` of word TF-IDF (1–2 grams, 10k) and character TF-IDF (2–5 grams, 20k), fitted on train only. Models: Logistic Regression, Calibrated Linear SVM, Multinomial Naive Bayes on the full TF-IDF; KNN, Decision Tree and Random Forest on a 100-dimension SVD (PCA-style) version; K-Means and Hierarchical clustering on toxic comments. The neural-network stage (Exp 11–13) is not part of this project.

## 4. Results — single split, seed 42 (`04_model_comparison.ipynb`)
| Model | Features | ROC-AUC | Macro F1 |
|---|---|---|---|
| Logistic Regression | TF-IDF (30k) | 0.9846 | 0.5557 |
| Calibrated Linear SVM | TF-IDF (30k) | 0.9812 | 0.5770 |
| Multinomial Naive Bayes | TF-IDF (30k) | 0.9662 | 0.4580 |
| Random Forest (40 trees) | SVD (100) | 0.9310 | 0.2798 |
| KNN (k=31, 40k subset) | SVD (100) | 0.8812 | 0.3092 |
| Decision Tree (depth=4) | SVD (100) | 0.8715 | 0.2224 |

KNN was fitted on a 40,000-comment subset and scored on 10,000 validation comments for speed. Models on only 100 SVD numbers lose a lot of word-level detail, which explains the gap to the TF-IDF linear models.

## 5. Tuning and benchmark — 3 random splits (`05_tuning_and_benchmark.ipynb`)
Tuning used only training data (inner split): Logistic Regression `C=4`, SVM `C=0.1`.

| Model | ROC-AUC (mean ± std) | Macro F1 (mean ± std) | Train time (s) |
|---|---|---|---|
| Calibrated SVM (C=0.1) | 0.9856 +/- 0.0011 | 0.6014 +/- 0.0026 | 106 |
| Logistic Regression (C=4.0) | 0.9838 +/- 0.0013 | 0.5910 +/- 0.0108 | 71 |
| Naive Bayes (alpha=0.1) | 0.9646 +/- 0.0019 | 0.4556 +/- 0.0004 | 5 |

Best: **Calibrated SVM, 0.9856**. Kaggle winner: **0.9885** (private leaderboard); a plain TF-IDF + Logistic Regression baseline was reported around 0.976 on the public leaderboard. So we beat the simple baseline and sit **0.0029** below the winner on a rough, not exactly comparable measure.
*What helped:* tuning `C` (SVM overtook Logistic Regression only after tuning) and a leakage-free split. *Honest gap:* the winners used large neural-network ensembles with pretrained embeddings; this project deliberately stays with fast, explainable linear models. Decision Tree, KNN and Random Forest were not repeated over 3 splits because they were clearly behind and costly to repeat.
Note: the capstone notebook uses the course-specified settings (SVM `C=0.5`, Logistic Regression `C=1.0`), where Logistic Regression scored higher (0.9846 vs 0.9812) and is the model saved for the app.

## 6. Business translation
With the tuned SVM and the 3-tier rule (split 1): **75.1%** of toxic comments do not go live (60.8% removed automatically, the rest reviewed), only **0.24%** of clean comments are wrongly removed, and humans must read about **2.1%** of all comments. With the capstone's Logistic Regression, **29.0%** of toxic comments are still published, 2.4% go to review, and only 0.1% of clean comments are wrongly removed. The threshold table in the capstone notebook shows the trade-off: lowering the PUBLISH threshold from 0.40 to 0.10 cuts missed toxic comments from 29.0% to 9.0%, but 6.7% of clean comments then need a human look instead of 0.9%. The right threshold is a business decision based on moderator cost.

## 7. Does the character branch help? (capstone, Sections 6–7)
* **Overall: yes.** Logistic Regression ROC-AUC **0.9751 (word-only) -> 0.9846 (hybrid)**.
* **Obfuscation specifically: not proven.** `You are an idiot` scores 1.00, but `1d10t` scores 0.74 and `i.d.i.o.t` 0.68 (REVIEW, not REMOVE), and a word-only model scored `1d10t` *higher* (0.88). The hybrid model still reacts to disguised insults, but weakly.
* **Explainability:** top Logistic Regression features per label are sensible (e.g. `threat`: die, kill, kill you; `identity_hate`: group words and slurs; character fragments like `uck`, `fuc` appear for profanity).

## 8. Clustering (`06_clustering.ipynb`)
K-Means on 13,004 toxic comments, K=4 chosen by silhouette (0.075, weak). One clear cluster is profanity-heavy (98.8% `obscene`, 34% `severe_toxic`); the others are mostly generic "you / your" text. Hierarchical clustering (1,000-comment sample) agrees only moderately (Adjusted Rand Index 0.469). Clustering is exploratory and not used in the deployed pipeline.

## 9. Deployment
`app/app.py` (Streamlit; runs locally, and is deploy-ready for Streamlit Community Cloud, with both ~1 MB model artifacts committed) loads `models/tfidf_union.pkl` and `models/best_model.pkl`, shows the decision badge and a six-bar probability chart. Checked in a headless run: "great work" -> PUBLISH, "You are an idiot" -> REMOVE, "You are an 1d10t" -> REVIEW, "This is complete bullshit." -> REMOVE, "I will find you and make you regret this" -> PUBLISH.

## 10. Limitations
* The bag-of-n-grams model has no sentence understanding: **"I will find you and make you regret this" is PUBLISHED** (threat probability 0.23), and "People like you from that country ruin everything" also goes live. Implicit threats and hate without profanity are the main failure.
* Very rare labels (`threat`, `identity_hate`) are weaker and noisier; Macro F1 at 0.5 is only about 0.56–0.60.
* Short sentences with "you are" can be over-scored ("You are a wonderful person" -> 0.52).
* Possible bias from training data: identity words appear mostly in toxic comments, so such words may be over-penalised.
* Thresholds (0.40 / 0.60 / 0.75) are hand-chosen; a real platform should tune them with real costs. Validation is on `train.csv`, not Kaggle's hidden test set.

## 11. References
1. Kaggle, *Toxic Comment Classification Challenge* (Jigsaw / Conversation AI), https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge — dataset (CC0) and metric.
2. NYC Data Science Academy, "Jigsaw's Text Classification Challenge" (student blog, 2018) — Logistic Regression baseline score.
3. Public write-up of the competition leaderboard (nsu.ru slides) — winning private score 0.9885.
4. `train.csv` obtained from the public GitHub mirror `iampukar/toxic-comments-classification` (file verified to have 159,571 rows and the official label counts).
