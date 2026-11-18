# PROJECT PROPOSAL — CSE303

**Name / Roll Number:** D. Abhiram / AP24110010593
**Project:** Intelligent Toxicity Moderation Platform

## Problem statement
Classify online comments for six kinds of toxicity at once — `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate` (a multi-label problem). The model scores every comment so humans only read the risky ones.

## Business framing
The decision: **publish, send to a human moderator, or remove**. The errors are asymmetric. A *false positive* (clean comment removed) is unfair censorship; a *false negative* leaves hate speech or threats visible. Missing a `threat` is worse than missing a mild insult, so the system uses three tiers: PUBLISH (max probability < 0.40), REVIEW (0.40–0.75, human), REMOVE (> 0.75, or `threat` / `severe_toxic` > 0.60).

## Related work
1. Kaggle *Toxic Comment Classification Challenge* (Jigsaw / Conversation AI, 2017–18), scored by mean column-wise ROC-AUC. The winning team reached about **0.9885** privately (0.9890 public) with large neural-network ensembles.
2. NYC Data Science Academy blog (Mar 2018): a TF-IDF + Logistic Regression baseline scored about **0.976** on the public leaderboard.

**Benchmark:** match the ~0.976 linear baseline and report the honest gap to 0.9885. I validate on a split of `train.csv`, not Kaggle's hidden test set.

## Dataset
Jigsaw `train.csv` — 159,571 Wikipedia comments, 6 binary labels, CC0 licence. Rare labels (`threat`, `identity_hate`) make it a real imbalance problem; TF-IDF + linear models train on a laptop.

## Planned approach
Follow the course ladder: Logistic Regression → KNN → Decision Tree → SVD (PCA idea) / chi-square analysis → Naive Bayes → SVM → Random Forest → K-Means + Hierarchical clustering → tune the top 2 → benchmark over 3 random splits (mean ± std). **Reviewer: please confirm that skipping the neural-network stage (Exp 11–13) is acceptable.** Features: hybrid word + character TF-IDF, fitted after the split. Metrics: column-averaged ROC-AUC and Macro F1, never plain accuracy.

## Deployment plan
A Streamlit app (deployed on Streamlit Community Cloud): type any comment, including disguised words like `1d10t`, click "Analyze Comment", and see the PUBLISH / REVIEW / REMOVE badge plus a bar chart of the six probabilities.

## Success criteria
Mean ROC-AUC (3 splits) at or above ~0.976; obfuscated insults scored close to plain versions; few toxic comments auto-published; the app runs with someone else's input; clearly stated failure cases (implicit threats); and an honest limitations section.

## "Make it yours" twist
(1) **Hybrid word + character TF-IDF** for disguised spellings, with an EDA check and a word-only vs hybrid ablation. (2) The **cost-aware 3-tier engine** evaluated with business numbers. (3) **Explainability:** top-weighted features per label.
