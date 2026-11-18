# Viva and Poster Day notes

Personal prep sheet — short answers in my own words, plus the numbers I will quote.
All scores are on a held-out 20% split of the 159,571 Jigsaw comments, not Kaggle's test set.

## 30-second pitch
I score every comment for six kinds of toxicity, then turn the six probabilities into one action: PUBLISH, REVIEW or REMOVE. A clean comment removed is unfair censorship; a toxic comment published is a threat left online. So I use a human REVIEW lane instead of one cut-off, and a lower bar for `threat` and `severe_toxic`. Best model is a calibrated linear SVM on word + character TF-IDF: ROC-AUC about 0.985 over three random splits, within 0.003 of the Kaggle winner, and 75% of toxic comments do not go live.

## 1. Why the split comes before the vectorizer (leakage)
I split the comments first, then `fit` the TF-IDF on the training text only and `transform` the validation text with those frozen words and IDF weights.

If I fitted the vectorizer on all 159,571 comments, the vocabulary and the IDF values of the validation comments would be part of the model's feature space. The model would be scored on features that were partly built from the answers it is supposed to predict. That is leakage: my validation score would look better than deployment, and the deployed model would see words during training statistics that it should not have. It is subtler than leaking labels — TF-IDF is unsupervised — but IDF is still a statistic of the evaluation text, and with 20,000 character n-grams rare fragments can encode near-unique comments.

The same rule applies to every fitted step: chi-square is computed on training data, SVD is fitted on the training matrix and only applied to validation, and the deployed `tfidf_union.pkl` is fitted on the 80% training split. That is why the validation numbers describe exactly the model in the app.

## 2. What FeatureUnion, OneVsRestClassifier and calibration do
**FeatureUnion** runs two vectorizers in parallel and glues their outputs side by side: word TF-IDF (1–2 grams, 10k columns) learns meaning (`idiot`, `kill you`), character TF-IDF (`char_wb`, 2–5 grams, 20k columns) learns spelling fragments (`1d10t`, `i.d.i.o.t`, `uck`). The result is one sparse matrix with 30,000 columns and the same 127,656 rows. One model sees both views; I do not train two models and average them.

**OneVsRestClassifier** trains six independent binary problems, one per label: "toxic or not", "threat or not", and so on. A comment can trigger several labels at once (`toxic` + `obscene` + `insult`), so this is multi-label, not multi-class. Each label gets its own decision boundary and its own coefficients, which is also why the six probabilities do not sum to 1. With a linear model it is cheap: six sets of weights over the same 30k features.

**Calibration** turns decision scores into probabilities. `LinearSVC` only gives a margin (distance from the boundary), not a probability, so the thresholds 0.40 / 0.60 / 0.75 would be meaningless numbers. `CalibratedClassifierCV(LinearSVC, cv=3)` fits a sigmoid on held-out folds, mapping margins to 0–1. After calibration a 0.80 means roughly "80% of comments looking like this are positive", which is what a moderation policy needs. It also makes the SVM's probabilities comparable with logistic regression's.

## 3. Why ROC-AUC and Macro F1 instead of accuracy
89.8% of comments have no label, so a model that always says "clean" is 89.8% accurate and completely useless. Accuracy hides the rare classes I care about most (`threat` is 0.30% of the data, `identity_hate` 0.87%).

**ROC-AUC** asks: if I pick one toxic and one clean comment, how often does the toxic one get the higher score? It is threshold-free, so it measures ranking quality independently of the 0.40/0.75 policy, and column-averaging it matches the Kaggle metric. It is also the metric I can compare against the leaderboard.

**Macro F1** is the F1 computed per label and then averaged with equal weight, so `threat` counts as much as `toxic`. It scores the actual yes/no decisions at a threshold and punishes both false positives and false negatives. That is why it is much lower (about 0.56–0.60) than ROC-AUC: ranking is easy, deciding is hard on rare labels. I report both, and never plain accuracy.

## 4. The 3-tier rule
| Decision | Rule | Reason |
|---|---|---|
| PUBLISH | max probability < 0.40 | confident enough to go live automatically |
| REVIEW | 0.40 ≤ max probability ≤ 0.75 | uncertain, a human reads it |
| REMOVE | max probability > 0.75, **or** `threat` / `severe_toxic` > 0.60 | confident, or a critical label with a lower bar |

The point is asymmetric cost. A false positive censors a normal user; a false negative leaves hate speech or a threat visible. A missed threat is worse than a missed mild insult, so `threat` and `severe_toxic` get a lower bar (0.60) even when the overall maximum is below 0.75. The REVIEW lane is the compromise: instead of forcing a binary decision, uncertain comments go to a human, which keeps moderator workload bounded.

Business numbers with the tuned SVM: 75.1% of toxic comments do not go live (60.8% auto-removed, the rest reviewed), only 0.24% of clean comments are wrongly removed, and humans read about 2.1% of all comments. Lowering PUBLISH from 0.40 to 0.10 cuts missed toxic comments from 29.0% to 9.0%, but human review rises from 0.9% to 6.7% of clean comments — the right threshold is a business decision.

## 5. Honest weak spots (say these before the examiner finds them)
**The obfuscation effect is small.** Overall the character branch clearly helps: word-only ROC-AUC 0.9750 → hybrid 0.9846. But the story I wanted — "character n-grams catch disguised insults" — is not proven. `You are an idiot` scores 1.00, `1d10t` scores 0.74 and `i.d.i.o.t` 0.68, both only REVIEW; and the word-only model scored `1d10t` *higher* (0.88). The EDA explains why: my regex patterns for disguise are about as common in clean comments as in toxic ones (letter-symbol-letter 1.48% toxic vs 1.73% clean), so there is little disguise-specific signal to learn. The likely benefit of the character branch is general robustness to typos and word fragments, not decoding leetspeak. I report this as a negative result rather than claiming the twist worked.

**`"I will find you and make you regret this"` gets PUBLISHED** (threat probability 0.23). This is a real threat with no threat vocabulary: `find`, `you`, `regret` are ordinary words, and a bag-of-n-grams has no syntax or compositionality. "People like you from that country ruin everything" also goes live. Implicit threats and hate without profanity are the main failure mode. I demonstrate this failure in the app during Poster Day on purpose — a model that only catches swear words is exactly the trap this project is about.

Other limitations I will admit: Macro F1 ~0.56–0.60, rare labels are noisier; "You are a wonderful person" is over-scored (0.52 → REVIEW); identity words appear mostly in toxic training comments, so the model may over-penalise them (bias); thresholds are hand-chosen, a real platform should tune them with real costs; validation is on a split of `train.csv`, not Kaggle's hidden test set; KNN and the tree/RF models use only 100 SVD dimensions, so their comparison is not purely "algorithm vs algorithm".

## 6. Numbers to have ready
| Model | ROC-AUC | Macro F1 |
|---|---|---|
| Logistic Regression, single split | 0.9846 | 0.5557 |
| Calibrated Linear SVM, single split | 0.9812 | 0.5770 |
| Naive Bayes | 0.9662 | 0.4580 |
| Random Forest (SVD 100) | 0.9310 | 0.2798 |
| KNN (SVD 100, 40k subset) | 0.8812 | 0.3092 |
| Decision Tree (SVD 100) | 0.8715 | 0.2224 |

Three random splits (tuned): calibrated SVM **0.9856 ± 0.0011** (Macro F1 0.6015), Logistic Regression **0.9838 ± 0.0013** (0.5910), Naive Bayes 0.9646 (0.4555). Kaggle winner 0.9885 private. Gap ≈ 0.003, on a not-exactly-comparable measure.

## 7. Likely questions and short answers
- **Why no neural network?** Scope and hardware: the course ladder ends at classical models; the point is a leak-free, explainable, deployable pipeline. The cost is visible: about 0.003 ROC-AUC below the winning transformer/embedding ensembles. I benchmark against that gap honestly instead of hiding it.
- **Why is Macro F1 only ~0.56 when AUC is 0.98?** AUC is ranking, F1 is thresholded decisions on six labels where five are rare. A single 0.5 cut-off is deliberately conservative; the tiered engine is tuned for business cost, not F1.
- **Why calibrate the SVM?** Linear SVC has no `predict_proba`; thresholds need probabilities. Sigmoid calibration on held-out folds gives them, and lets me compare SVM and LR on the same scale.
- **Why character n-grams 2–5?** 2 captures short fragments (`uk`), 5 captures word-ish pieces; more would explode the 20k budget and slow training.
- **Why stratify only on `toxic`?** It is the only label large enough to stratify reliably; the rarer labels keep approximately the same rate (checked in notebook 02).
- **Why is the saved app model Logistic Regression if SVM wins the benchmark?** The capstone uses course-specified settings (`C=1.0`, SVM `C=0.5`), where LR scores higher (0.9846 vs 0.9812). The tuned SVM (`C=0.1`) wins over three splits. Both are reported; the app ships the notebook's best single-split model so its validation numbers describe it exactly.
- **What does the clustering add?** Exploration only, not deployed: K=4 chosen by silhouette 0.076 (weak), one clear profanity-heavy cluster (98.8% obscene), moderate agreement with hierarchical clustering (ARI 0.445). It shows the labels are not the only structure in the data.
- **Is 0.985 "better than Kaggle"?** No. Different evaluation: my 20% split of `train.csv` versus the hidden test set. I say "within 0.003 of the winner on a rough comparison", never "beat the leaderboard".

## 8. Poster Day demo script
1. `great work` → PUBLISH (no false alarm).
2. `You are an idiot` → REMOVE, show the six bars and the top features (`idiot`, `stupid`).
3. `You are an 1d10t` → REVIEW — honest: the disguise weakens the score instead of being decoded.
4. `I will find you and make you regret this` → **PUBLISH** — the failure I want people to see; explain why bag-of-words cannot represent an implicit threat.
5. Close with the 3-tier rule and the business numbers: 75% of toxicity blocked, 0.24% of clean comments wrongly removed.

## 9. One-line definitions to memorise
- **Leakage:** fitting any preprocessing on data that will later be used for evaluation.
- **FeatureUnion:** parallel feature extractors, horizontally stacked into one matrix.
- **OneVsRest:** one binary classifier per label, probabilities independent.
- **Calibration:** mapping model scores to frequencies that behave like probabilities.
- **ROC-AUC:** probability a positive is ranked above a negative; threshold-free.
- **Macro F1:** per-label F1 averaged equally, so rare labels matter.
- **3-tier rule:** PUBLISH / REVIEW / REMOVE with a lower bar for critical labels.
