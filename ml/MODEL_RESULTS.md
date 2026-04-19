# DepScreen — Complete Model Results & ML Stack

> Single authoritative reference for all model metrics, per-class performance, ablation results, data provenance, and ML/DL stack details. Updated 2026-04-19.

---

## 1. Production Model: 3-Model Soft-Vote Ensemble

### Architecture

| Component | Model | Parameters | Tokenizer | Batch Size | Notes |
|-----------|-------|-----------|-----------|-----------|-------|
| Model A | DAPT'd DistilBERT | 66M | WordPiece | 16 | Domain-adapted on 39K Reddit MH posts |
| Model B | RoBERTa-base | 125M | BPE (GPT-2) | 16 | Different subword segmentation |
| Model C | DeBERTa-base (v1) | 139M | SentencePiece | 4 | Disentangled attention; batch=4 for MPS OOM |

**Inference pipeline:**
1. Each model tokenizes the input sentence independently
2. Each produces a softmax probability distribution over 11 classes
3. Probabilities are averaged (soft-vote)
4. Per-class thresholds adjust the decision boundary
5. **Safety override**: if P(SUICIDAL_THOUGHTS) ≥ 0.15, always predict SUICIDAL_THOUGHTS

### Shared Training Config

| Parameter | Value |
|-----------|-------|
| Pooling | Mean (average all token embeddings, attention-masked) |
| Learning rate | 3e-5 |
| Epochs | 7 (best checkpoint by val Micro-F1) |
| Max length | 128 tokens |
| Loss | CrossEntropyLoss (effective-number weights β=0.999 + label smoothing ε=0.1) |
| Optimizer | AdamW + linear warmup (10% of steps) + linear decay |
| Gradient clipping | max norm 1.0 |
| Device | Apple Silicon MPS |
| Evaluation | 5-fold MultilabelStratifiedKFold at post level (iterstrat) |

### Production Training

| Parameter | Value |
|-----------|-------|
| Training data | 1,792 samples (cleaned train + val + augmented) |
| Validation holdout | None — CV already provided performance estimate |
| Seed | 42 |

---

## 2. Ensemble CV Results (5-Fold)

### Aggregate Metrics

| Metric | Raw Ensemble | Threshold-Tuned |
|--------|-------------|-----------------|
| **Micro-F1** | **0.813 ± 0.010** | **0.820** |
| **Macro-F1** | **0.770 ± 0.017** | **0.792** |
| Accuracy | 0.813 ± 0.010 | 0.820 |

> Threshold-tuned metrics have slight optimistic bias (tuned on eval data, not nested CV). True performance is between raw and tuned values.

### Per-Fold Ensemble Micro-F1

| Fold | Micro-F1 | Macro-F1 | Val Samples |
|------|----------|----------|-------------|
| 1 | 0.803 | 0.790 | 356 |
| 2 | 0.828 | 0.785 | 296 |
| 3 | 0.802 | 0.761 | 318 |
| 4 | 0.821 | 0.744 | 313 |
| 5 | 0.808 | 0.769 | 313 |

### Per-Class F1 — Raw Ensemble (5-fold mean)

| Symptom | F1 | Precision | Recall | Avg Support/Fold |
|---------|-----|-----------|--------|-----------------|
| SUICIDAL_THOUGHTS | 0.973 | 0.957 | 0.989 | 36.4 |
| WORTHLESSNESS | 0.906 | 0.894 | 0.919 | 62.0 |
| SLEEP_ISSUES | 0.879 | 0.807 | 0.970 | 20.4 |
| FATIGUE | 0.892 | 0.877 | 0.912 | 25.2 |
| DEPRESSED_MOOD | 0.878 | 0.848 | 0.913 | 63.0 |
| ANHEDONIA | 0.831 | 0.787 | 0.883 | 22.4 |
| APPETITE_CHANGE | 0.810 | 0.721 | 0.933 | 8.8 |
| PSYCHOMOTOR | 0.672 | 0.585 | 0.805 | 6.2 |
| COGNITIVE_ISSUES | 0.718 | 0.619 | 0.865 | 10.2 |
| SPECIAL_CASE | 0.547 | 0.575 | 0.540 | 13.4 |
| NO_SYMPTOM | 0.362 | 0.528 | 0.278 | 51.2 |

### Per-Class F1 — Threshold-Tuned (aggregated across all folds)

| Symptom | F1 | Precision | Recall | Total Support |
|---------|-----|-----------|--------|--------------|
| SUICIDAL_THOUGHTS | **0.978** | 0.973 | 0.984 | 182 |
| WORTHLESSNESS | **0.917** | 0.927 | 0.906 | 310 |
| SLEEP_ISSUES | **0.893** | 0.850 | 0.941 | 102 |
| FATIGUE | **0.891** | 0.909 | 0.873 | 126 |
| DEPRESSED_MOOD | **0.874** | 0.835 | 0.917 | 315 |
| ANHEDONIA | **0.838** | 0.821 | 0.857 | 112 |
| APPETITE_CHANGE | **0.825** | 0.755 | 0.909 | 44 |
| PSYCHOMOTOR | **0.727** | 0.686 | 0.774 | 31 |
| COGNITIVE_ISSUES | **0.692** | 0.561 | 0.902 | 51 |
| SPECIAL_CASE | **0.569** | 0.673 | 0.493 | 67 |
| NO_SYMPTOM | **0.508** | 0.591 | 0.445 | 256 |

### Per-Class Thresholds (Production)

| Class | Threshold |
|-------|-----------|
| DEPRESSED_MOOD | 0.25 |
| ANHEDONIA | 0.45 |
| APPETITE_CHANGE | 0.50 |
| SLEEP_ISSUES | 0.55 |
| PSYCHOMOTOR | 0.50 |
| FATIGUE | 0.60 |
| WORTHLESSNESS | 0.40 |
| COGNITIVE_ISSUES | 0.15 |
| SUICIDAL_THOUGHTS | 0.05 (lowered from 0.40 — safety-critical) |
| SPECIAL_CASE | 0.55 |
| NO_SYMPTOM | 0.15 |

---

## 3. Individual Model CV Results (5-Fold, cleaned data)

### DAPT'd DistilBERT

| Metric | Value |
|--------|-------|
| Micro-F1 | 0.786 ± 0.013 |
| Macro-F1 | 0.763 ± 0.027 |

| Class | F1 (μ±σ) | Prec (μ±σ) | Rec (μ±σ) | Support |
|-------|----------|------------|-----------|---------|
| DEPRESSED_MOOD | 0.834 ± 0.018 | 0.831 ± 0.019 | 0.838 ± 0.037 | 63.0 |
| ANHEDONIA | 0.834 ± 0.044 | 0.896 ± 0.037 | 0.781 ± 0.055 | 22.4 |
| APPETITE_CHANGE | 0.827 ± 0.101 | 0.751 ± 0.101 | 0.933 ± 0.133 | 8.8 |
| SLEEP_ISSUES | 0.879 ± 0.054 | 0.830 ± 0.082 | 0.942 ± 0.053 | 20.4 |
| PSYCHOMOTOR | 0.708 ± 0.128 | 0.766 ± 0.161 | 0.681 ± 0.171 | 6.2 |
| FATIGUE | 0.875 ± 0.026 | 0.886 ± 0.041 | 0.871 ± 0.067 | 25.2 |
| WORTHLESSNESS | 0.859 ± 0.039 | 0.878 ± 0.047 | 0.844 ± 0.053 | 62.0 |
| COGNITIVE_ISSUES | 0.661 ± 0.124 | 0.586 ± 0.155 | 0.780 ± 0.115 | 10.2 |
| SUICIDAL_THOUGHTS | 0.964 ± 0.011 | 0.957 ± 0.032 | 0.973 ± 0.016 | 36.4 |
| SPECIAL_CASE | 0.484 ± 0.075 | 0.722 ± 0.101 | 0.371 ± 0.078 | 13.4 |
| NO_SYMPTOM | 0.468 ± 0.110 | 0.468 ± 0.128 | 0.474 ± 0.097 | 51.2 |

### RoBERTa-base

| Metric | Value |
|--------|-------|
| Micro-F1 | 0.811 ± 0.016 |
| Macro-F1 | 0.767 ± 0.033 |

| Class | F1 (μ±σ) | Prec (μ±σ) | Rec (μ±σ) | Support |
|-------|----------|------------|-----------|---------|
| DEPRESSED_MOOD | 0.876 ± 0.024 | 0.845 ± 0.045 | 0.911 ± 0.020 | 63.0 |
| ANHEDONIA | 0.852 ± 0.037 | 0.803 ± 0.040 | 0.911 ± 0.058 | 22.4 |
| APPETITE_CHANGE | 0.813 ± 0.068 | 0.727 ± 0.095 | 0.936 ± 0.088 | 8.8 |
| SLEEP_ISSUES | 0.872 ± 0.039 | 0.816 ± 0.072 | 0.943 ± 0.034 | 20.4 |
| PSYCHOMOTOR | 0.617 ± 0.198 | 0.588 ± 0.124 | 0.671 ± 0.282 | 6.2 |
| FATIGUE | 0.893 ± 0.042 | 0.869 ± 0.047 | 0.920 ± 0.055 | 25.2 |
| WORTHLESSNESS | 0.897 ± 0.022 | 0.897 ± 0.045 | 0.899 ± 0.026 | 62.0 |
| COGNITIVE_ISSUES | 0.684 ± 0.091 | 0.601 ± 0.108 | 0.802 ± 0.061 | 10.2 |
| SUICIDAL_THOUGHTS | 0.957 ± 0.015 | 0.937 ± 0.020 | 0.979 ± 0.020 | 36.4 |
| SPECIAL_CASE | 0.552 ± 0.054 | 0.568 ± 0.075 | 0.553 ± 0.096 | 13.4 |
| NO_SYMPTOM | 0.426 ± 0.149 | 0.585 ± 0.137 | 0.341 ± 0.132 | 51.2 |

### DeBERTa-base

| Metric | Value |
|--------|-------|
| Micro-F1 | 0.804 ± 0.010 |
| Macro-F1 | 0.766 ± 0.023 |

| Class | F1 (μ±σ) | Prec (μ±σ) | Rec (μ±σ) | Support |
|-------|----------|------------|-----------|---------|
| DEPRESSED_MOOD | 0.865 ± 0.040 | 0.841 ± 0.060 | 0.894 ± 0.043 | 63.0 |
| ANHEDONIA | 0.782 ± 0.034 | 0.725 ± 0.078 | 0.865 ± 0.094 | 22.4 |
| APPETITE_CHANGE | 0.807 ± 0.076 | 0.730 ± 0.097 | 0.911 ± 0.084 | 8.8 |
| SLEEP_ISSUES | 0.877 ± 0.057 | 0.810 ± 0.075 | 0.960 ± 0.041 | 20.4 |
| PSYCHOMOTOR | 0.675 ± 0.157 | 0.624 ± 0.149 | 0.738 ± 0.173 | 6.2 |
| FATIGUE | 0.888 ± 0.040 | 0.862 ± 0.050 | 0.920 ± 0.055 | 25.2 |
| WORTHLESSNESS | 0.900 ± 0.040 | 0.896 ± 0.070 | 0.906 ± 0.028 | 62.0 |
| COGNITIVE_ISSUES | 0.675 ± 0.104 | 0.568 ± 0.106 | 0.838 ± 0.109 | 10.2 |
| SUICIDAL_THOUGHTS | 0.962 ± 0.009 | 0.952 ± 0.019 | 0.973 ± 0.016 | 36.4 |
| SPECIAL_CASE | 0.592 ± 0.081 | 0.656 ± 0.097 | 0.555 ± 0.113 | 13.4 |
| NO_SYMPTOM | 0.402 ± 0.136 | 0.591 ± 0.125 | 0.319 ± 0.126 | 51.2 |

---

## 4. Improvement Journey — Full Ablation

| # | Version | Change | Micro-F1 | Macro-F1 | Δ Micro |
|---|---------|--------|----------|----------|---------|
| 0 | baseline_v1 | DistilBERT, single split (test set) | 0.696 | 0.652 | — |
| 1 | baseline_v1 (CV) | Same, 5-fold CV | 0.677 ± 0.014 | 0.665 ± 0.008 | baseline |
| 2 | v2_dapt_base | + DAPT on 39K Reddit MH posts | 0.686 ± 0.005 | 0.675 ± 0.009 | +0.9% |
| 3 | v3_distilled | + Knowledge distillation (Gemini) | 0.687 ± 0.007 | 0.672 ± 0.010 | +1.0% |
| 4 | v4_loss_tuned | + Effective-num weights + label smoothing | 0.686 ± 0.011 | 0.671 ± 0.011 | +0.9% |
| 5 | v5_augmented | + LLM augmentation (196 samples) | 0.698 ± 0.016 | 0.686 ± 0.014 | +2.1% |
| 6 | **v6_cleaned** | **+ Data cleaning (confident learning)** | **0.791 ± 0.012** | **0.755 ± 0.015** | **+11.4%** |
| 7 | v7_mean_pooling | + Mean pooling, lr=3e-5, 7 epochs | 0.796 ± 0.010 | 0.762 ± 0.012 | +11.9% |
| 8 | **v8_ensemble** | **+ 3-model ensemble** | **0.813 ± 0.010** | **0.770 ± 0.017** | **+13.6%** |
| — | v8 + thresholds | + Per-class thresholds | 0.820 | 0.792 | +14.3%* |

*Threshold-tuned metrics have optimistic bias (not nested CV).

### Impact Ranking

| Rank | Technique | Micro-F1 Impact |
|------|-----------|-----------------|
| 1 | **Data cleaning** (conflict resolution, dedup, confident learning) | **+11.4%** |
| 2 | **Ensemble** (3 architecturally diverse models) | +2.1% |
| 3 | Augmentation (LLM paraphrasing, similarity-filtered) | +1.2% |
| 4 | DAPT (domain-adaptive pre-training on Reddit MH text) | +0.9% |
| 5 | Mean pooling + lr=3e-5 + 7 epochs | +0.5% |
| 6 | Loss tuning (effective-number weights + label smoothing) | marginal |
| 7 | Knowledge distillation (Gemini soft labels) | marginal (but +193% on NO_SYMPTOM) |

### What Didn't Work

| Technique | Result | Why |
|-----------|--------|-----|
| LLRD + FGM + R-Drop regularization | −2% Micro-F1 | Overcorrected on 1,418 samples |
| Aggressive data augmentation (>200 synthetic) | Degraded | Distribution mismatch |
| DL-validated augmentation for PSYCHOMOTOR | No gain | Self-reinforcing loop |
| SWA (Stochastic Weight Averaging) | No gain | Best checkpoint always better |
| CLS+Mean concatenated pooling | No gain | Too many parameters for data size |
| Focal loss (γ=2.0) | High variance | Catastrophic fold 4 (macro=0.602) |

---

## 5. Data Provenance

### Original Dataset: ReDSM5 (CIKM 2025)

| Stat | Value |
|------|-------|
| Source | Pérez et al., "ReDSM5: A Reddit Dataset for DSM-5 Depression Detection" |
| Posts | 1,484 anonymized Reddit posts from mental health subreddits |
| Annotations | 2,058 sentence-level labels by a licensed psychologist |
| Classes | 11 (9 DSM-5 symptoms + SPECIAL_CASE + NO_SYMPTOM) |
| Multi-annotated | 65 sentences have 2+ annotations |

### Data Cleaning (v6 — the breakthrough)

| Step | Action | Samples Affected |
|------|--------|-----------------|
| Conflict resolution | Clinical salience hierarchy for multi-annotator disagreements | 53 sentences |
| Deduplication | Exact + near-duplicate removal | 20 sentences |
| Confident learning | Cross-validated model predictions → manual review | 66 relabeled, 96 removed |
| Manual fixes | Egregious misannotations found during review | 9 sentences |
| **Key finding** | **189/324 NO_SYMPTOM labels were actually symptoms** | — |

Confident learning thresholds: high=0.85, medium=0.70.

### Production Training Data Distribution

| Class | Train | Val | Aug | Total | Effective Weight |
|-------|-------|-----|-----|-------|-----------------|
| DEPRESSED_MOOD | 279 | 36 | 0 | 315 | 0.462 |
| WORTHLESSNESS | 280 | 30 | 0 | 310 | 0.460 |
| NO_SYMPTOM | 220 | 36 | 0 | 256 | 0.586 |
| SUICIDAL_THOUGHTS | 162 | 20 | 0 | 182 | 0.796 |
| FATIGUE | 114 | 12 | 0 | 126 | 1.131 |
| ANHEDONIA | 105 | 7 | 0 | 112 | 1.228 |
| SLEEP_ISSUES | 89 | 13 | 0 | 102 | 1.448 |
| SPECIAL_CASE | 58 | 9 | 0 | 67 | 2.223 |
| COGNITIVE_ISSUES | 43 | 8 | 0 | 51 | 2.998 |
| APPETITE_CHANGE | 41 | 3 | 0 | 44 | 3.144 |
| PSYCHOMOTOR | 27 | 4 | 0 | 31 | 4.774 |
| **Total** | **1,418** | **178** | **196** | **1,792** | — |

### Augmentation (196 samples)

| Class | Original | Augmented | Combined |
|-------|----------|-----------|----------|
| SPECIAL_CASE | 76 | 63 | 139 |
| APPETITE_CHANGE | 41 | 62 | 103 |
| COGNITIVE_ISSUES | 51 | 31 | 82 |
| SUICIDAL_THOUGHTS | 155 | 10 | 165 |
| PSYCHOMOTOR | 30 | 6 | 36 |

- Generator: Gemini 2.5 Flash, 5 paraphrases/sentence, temperature=0.8
- Quality filter: sentence-transformers cosine similarity ∈ [0.70, 0.95]
- PSYCHOMOTOR low pass rate (7%) — original sentences too specific/short

### DAPT Corpus

| Stat | Value |
|------|-------|
| Source | 15 mental health subreddits (r/depression, r/anxiety, r/SuicideWatch, etc.) |
| Raw posts | 12,178 (Reddit public JSON API) |
| After validation | 39,206 chunks (deduped, spam-filtered, English-only) |
| Leaked chunks removed | 8 (overlap with ReDSM5 train data) |
| Pre-DAPT perplexity | 16.90 |
| Post-DAPT perplexity | 7.59 (−55.1%) |
| Epochs | 3, batch=32, lr=5e-5, MLM prob=15% |

### Knowledge Distillation

| Stat | Value |
|------|-------|
| Teacher model | gemini-3-flash-preview (Google AI Studio) |
| Pilot Cohen's Kappa | 0.664 (substantial agreement) |
| Manual disagreement analysis | 7/11 LLM more correct than human annotator |
| Valid soft labels | 1,498/1,591 (94.2%), 93 fell back to one-hot |
| Distillation loss | α=0.6 (60% hard, 40% soft), T=3.0 |
| Processing time | 4,095s total, 2.57s/sample |
| Per-class alpha override | Hard-only for PSYCHOMOTOR, COGNITIVE_ISSUES, SPECIAL_CASE |

---

## 6. Baseline Comparison

### Published ReDSM5 Baselines vs DepScreen

| Model | Micro-F1 | Δ vs BERT | Δ vs DepScreen |
|-------|----------|-----------|----------------|
| CNN (10 epochs) | 0.25 | −0.26 | −0.56 |
| SVM (TF-IDF) | 0.39 | −0.12 | −0.42 |
| BERT (fine-tuned) | 0.51 | — | −0.30 |
| LLaMA-3.2-1B (fine-tuned) | 0.54 | +0.03 | −0.27 |
| DepScreen DistilBERT (single split) | 0.696 | +0.19 | −0.12 |
| DepScreen MentalBERT (single split) | 0.684 | +0.17 | −0.13 |
| DepScreen DistilBERT (5-fold CV) | 0.677 ± 0.014 | +0.17 | −0.14 |
| **DepScreen Ensemble (5-fold CV)** | **0.813 ± 0.010** | **+0.30** | **—** |

### DepScreen DistilBERT vs MentalBERT (Single Split, Test Set)

| Metric | DistilBERT | MentalBERT |
|--------|-----------|-----------|
| Micro-F1 | **0.696** | 0.684 |
| Macro-F1 | **0.652** | 0.650 |
| Macro-Precision | **0.613** | 0.598 |
| Macro-Recall | 0.796 | **0.800** |

| Symptom | DistilBERT F1 | MentalBERT F1 | Winner |
|---------|-------------|-------------|--------|
| SUICIDAL_THOUGHTS | 0.950 | **0.974** | Mental |
| WORTHLESSNESS | 0.853 | 0.853 | Tie |
| FATIGUE | 0.824 | **0.842** | Mental |
| DEPRESSED_MOOD | **0.763** | 0.694 | Distil |
| SLEEP_ISSUES | **0.759** | 0.741 | Distil |
| ANHEDONIA | 0.686 | **0.824** | Mental |
| SPECIAL_CASE | 0.667 | **0.714** | Mental |
| COGNITIVE_ISSUES | **0.571** | 0.533 | Distil |
| APPETITE_CHANGE | 0.500 | **0.545** | Mental |
| PSYCHOMOTOR | **0.500** | 0.333 | Distil |
| NO_SYMPTOM | **0.103** | 0.091 | Distil |

---

## 7. Baseline Training History

### DistilBERT (5 epochs, single split)

| Epoch | Train Loss | Train Acc | Val Loss | Val Micro-F1 | Val Macro-F1 |
|-------|-----------|----------|----------|-------------|-------------|
| 1 | 2.332 | 19.3% | 1.866 | 0.573 | 0.518 |
| 2 | 1.312 | 61.7% | 0.975 | 0.660 | 0.621 |
| 3 | 0.814 | 69.4% | 0.883 | 0.676 | 0.664 |
| **4** | **0.626** | **73.5%** | **0.889** | **0.692** | **0.674** |
| 5 | 0.548 | 76.2% | 0.870 | 0.681 | 0.672 |

Best checkpoint: epoch 4 (by val Micro-F1).

### MentalBERT (5 epochs, single split)

| Epoch | Train Loss | Train Acc | Val Loss | Val Micro-F1 | Val Macro-F1 |
|-------|-----------|----------|----------|-------------|-------------|
| 1 | 2.177 | 26.8% | 1.442 | 0.632 | 0.630 |
| 2 | 1.114 | 64.6% | 0.845 | 0.681 | 0.641 |
| **3** | **0.677** | **73.9%** | **0.790** | **0.730** | **0.687** |
| 4 | 0.509 | 77.8% | 0.821 | 0.724 | 0.680 |
| 5 | 0.426 | 80.6% | 0.805 | 0.714 | 0.674 |

Best checkpoint: epoch 3 (by val Micro-F1).

---

## 8. Safety-Critical: SUICIDAL_THOUGHTS

| Metric | Baseline (single) | Ensemble (raw) | Ensemble (tuned) |
|--------|-------------------|---------------|-----------------|
| F1 | 0.895 ± 0.038 | 0.973 | 0.978 |
| Precision | 0.859 ± 0.045 | 0.957 | 0.973 |
| Recall | 0.939 ± 0.062 | 0.989 | 0.984 |

### Safety Override

- **Hard floor**: if raw ensemble P(SUICIDAL_THOUGHTS) ≥ 0.15, always predict SUICIDAL_THOUGHTS
- **Threshold**: lowered from 0.40 to 0.05 (CV-tuned value was too aggressive)
- **Bug caught**: original threshold caused "Sometimes I wonder if everyone would be better off without me" to be classified as DEPRESSED_MOOD instead of SUICIDAL_THOUGHTS
  - Raw probs: SUICIDAL=0.426, DEPRESSED=0.292
  - Old adjusted: DEPRESSED=0.042 (wins), SUICIDAL=0.026 (loses)
  - Fixed: safety override triggers at 0.426 ≥ 0.15

---

## 9. LLM Stack (Hybrid Tier Strategy)

### Model Routing

| Tier | Model | Tasks | Why |
|------|-------|-------|-----|
| **Pro** | gemini-3.1-pro-preview | Adversarial detection, explanation generation, secondary symptoms | Needs chain-of-thought reasoning |
| **Flash** | gemini-3-flash-preview | Chat, evidence validation, confidence calibration | Balanced quality/speed/cost |
| **Lite** | gemini-3.1-flash-lite-preview | Auto-title, utilities | Cheap and fast |

### Provider Config

| Setting | Value |
|---------|-------|
| Provider | Google AI Studio (paid tier, billing enabled) |
| Endpoint | `https://generativelanguage.googleapis.com/v1beta/openai/` |
| Client | OpenAI SDK (compatible endpoint) |
| Timeout | 120s (bumped for Pro reasoning latency) |
| Max retries | 3 (tenacity, exponential backoff) |

### LLM Verification Pipeline (per screening)

Each detected symptom triggers 3 parallel Gemini calls:
1. **Evidence validation** (Flash) — does the sentence actually support this symptom?
2. **Adversarial check** (Pro) — does rewording the sentence flip the prediction?
3. **Confidence calibration** (Flash) — is the model's confidence honest?

Only flags surviving all 3 checks reach the clinician.

### SafetyGuard (LLM output redaction)

Scans every LLM response before delivery for:
- Prescription/dosage advice ("take 50mg", "start with X mg")
- Diagnostic claims ("you have depression", "you are suffering from")
- Self-harm encouragement
- Anti-professional content

Matched phrases replaced with safe substitutes + Bahrain crisis disclaimer appended.

---

## 10. File Locations

```
ml/
├── models/
│   ├── MODEL_REGISTRY.md              # Full version history
│   ├── MODEL_RESULTS.md               # This file
│   ├── baseline_v1/                    # Frozen baseline (chmod 444)
│   │   ├── symptom_classifier.pt
│   │   ├── redsm5_metadata.json
│   │   └── evaluation_report.md
│   ├── v2_dapt_base/                   # DAPT'd DistilBERT encoder
│   │   ├── model.safetensors
│   │   └── dapt_metadata.json
│   └── v_production_ensemble/          # ⭐ PRODUCTION
│       ├── ensemble_metadata.json      # Thresholds, label map, CV perf
│       ├── dapt_distilbert/            # model.pt + tokenizer
│       ├── roberta/                    # model.pt + tokenizer
│       └── deberta/                    # model.pt + tokenizer
├── data/
│   └── redsm5/
│       ├── cleaned_v2/                 # Production training data
│       │   ├── train.csv (1,418)
│       │   ├── val.csv (178)
│       │   └── metadata.json
│       ├── augmented/                  # v1 augmentation (172 samples)
│       └── distilled/                  # Gemini soft labels
├── evaluation/
│   └── cv_results/
│       ├── ensemble_cv_results.json    # Full per-fold, per-model results
│       ├── cv_results_distilbert-base-uncased_5fold.json
│       ├── cv_results_._ml_models_v2_dapt_base_5fold.json
│       ├── cv_results_roberta-base_5fold.json
│       └── cv_results_microsoft_deberta-base_5fold.json
└── scripts/
    ├── ensemble_cv.py                  # 5-fold ensemble CV
    ├── train_production.py             # Final training on 100% data
    ├── train_redsm5_model.py           # SymptomClassifier + SymptomDataset
    ├── clean_training_data.py          # Conflict resolution + dedup
    ├── apply_confident_learning.py     # Model-predicted mislabel detection
    ├── distill_labels.py               # Gemini soft label generation
    ├── augment_rare_classes.py         # LLM paraphrasing
    ├── train_dapt.py                   # MLM pre-training
    └── preprocess_redsm5.py            # Data loading + label definitions
```

---

## 11. Caveats & Limitations

1. **Threshold-tuned metrics are optimistic** — tuned on eval data, not nested CV. True performance is between raw ensemble (0.813/0.770) and tuned (0.820/0.792).
2. **Val data was cleaned after ensemble CV** — 6 samples corrected post-hoc. Production models benefit from this.
3. **Rare-class variance is high** — PSYCHOMOTOR (6.2 samples/fold), COGNITIVE_ISSUES (10.2), SPECIAL_CASE (13.4) have wide per-fold std.
4. **Fold 4 consistently weakest for NO_SYMPTOM** — F1=0.12 vs mean=0.36. Likely a data partition effect.
5. **Reddit ≠ clinical text** — model trained on informal social media. Generalization to clinical intake language is unvalidated.
6. **English only** — no Arabic/multilingual evaluation.
7. **DeBERTa requires batch_size=4 on Apple Silicon MPS** — OOM with batch_size=16.
8. **Single split baseline (0.696) was slightly lucky** — CV baseline is 0.677 (−2%).
