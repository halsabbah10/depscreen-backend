# DepScreen — Model Evaluation Report

**Model**: distilbert-base-uncased
**Date**: 2026-04-11

## Overall Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.6901 |
| Micro-F1 | 0.6901 |
| Macro-F1 | 0.6595 |
| Micro-Precision | 0.6901 |
| Micro-Recall | 0.6901 |

## Per-Symptom Breakdown

| Symptom           |   Precision |   Recall |   F1-Score |   Support |
|:------------------|------------:|---------:|-----------:|----------:|
| DEPRESSED_MOOD    |      0.6905 |   0.8056 |     0.7436 |        36 |
| ANHEDONIA         |      0.75   |   0.7059 |     0.7273 |        17 |
| APPETITE_CHANGE   |      0.4444 |   1      |     0.6154 |         4 |
| SLEEP_ISSUES      |      0.6875 |   0.9167 |     0.7857 |        12 |
| PSYCHOMOTOR       |      0.3333 |   1      |     0.5    |         1 |
| FATIGUE           |      0.7778 |   0.875  |     0.8235 |         8 |
| WORTHLESSNESS     |      0.8    |   0.8571 |     0.8276 |        28 |
| COGNITIVE_ISSUES  |      0.4    |   1      |     0.5714 |         4 |
| SUICIDAL_THOUGHTS |      0.95   |   1      |     0.9744 |        19 |
| SPECIAL_CASE      |      0.5    |   0.7143 |     0.5882 |         7 |
| NO_SYMPTOM        |      0.3333 |   0.0571 |     0.0976 |        35 |

## Baseline Comparison

| Model                       |   Micro-F1 | Delta vs Ours   |
|:----------------------------|-----------:|:----------------|
| SVM (TF-IDF)                |     0.39   | +0.3001         |
| CNN (10 epochs)             |     0.25   | +0.4401         |
| BERT (fine-tuned)           |     0.51   | +0.1801         |
| LLaMA 3.2-1B (fine-tuned)   |     0.54   | +0.1501         |
| DepScreen DistilBERT (ours) |     0.6901 | —               |

## Post-Level Aggregation

| Metric | Value |
|--------|-------|
| Posts evaluated | 143 |
| Severity classification accuracy | 77.62% |
| Symptom count MAE | 0.27 |
| Avg true symptoms/post | 0.89 |
| Avg predicted symptoms/post | 1.06 |
| Avg symptom overlap/post | 0.76 |

## Error Analysis

Total misclassifications: 53 / 171

### Sample Errors (first 20)

| sentence_text                                                                                                                                                                                                                                                                                                                                       | true_label     | predicted_label   | post_id     |
|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------|:------------------|:------------|
| I try my best to keep myself optimistic but most days it doesn't work as I would hope to like it too, then again it's probably best to not stay to optimistic just to eventually let myself down because I'd over hyped myself.                                                                                                                     | DEPRESSED_MOOD | SPECIAL_CASE      | s_586_428   |
| I'm also quite pessimistic by nature, but hope still gets me every time.                                                                                                                                                                                                                                                                            | DEPRESSED_MOOD | NO_SYMPTOM        | s_3003_261  |
| I still experience sexual satisfaction with my partner but I have just accepted that it will be more difficult for me now.                                                                                                                                                                                                                          | NO_SYMPTOM     | ANHEDONIA         | s_2625_5    |
| Recently I've been going to bed quite late and almost at the point of dropping into sleep as soon as my head hits the pillow.                                                                                                                                                                                                                       | NO_SYMPTOM     | SLEEP_ISSUES      | s_1330_18   |
| And now I'm sad.                                                                                                                                                                                                                                                                                                                                    | NO_SYMPTOM     | DEPRESSED_MOOD    | s_18_228    |
| Also I have lived recklessly because I thought I didn't had much future.                                                                                                                                                                                                                                                                            | DEPRESSED_MOOD | WORTHLESSNESS     | s_993_83    |
| Increased my personal interest in the game.                                                                                                                                                                                                                                                                                                         | SPECIAL_CASE   | ANHEDONIA         | s_1728_52   |
| i want sex.                                                                                                                                                                                                                                                                                                                                         | NO_SYMPTOM     | ANHEDONIA         | s_2107_102  |
| I still feel a lot of guilt, but working with my dog and truly spending time with him doing things he wants to do and being there as his cheerleader as he learns new things has helped ease some of that guilt.                                                                                                                                    | WORTHLESSNESS  | NO_SYMPTOM        | s_1834_697  |
| When our brains can no longer process the sadness, anhedonia sets in.                                                                                                                                                                                                                                                                               | ANHEDONIA      | DEPRESSED_MOOD    | s_1625_393  |
| I just can't focus                                                                                                                                                                                                                                                                                                                                  | NO_SYMPTOM     | COGNITIVE_ISSUES  | s_1210_72   |
| Life is hell.                                                                                                                                                                                                                                                                                                                                       | DEPRESSED_MOOD | SUICIDAL_THOUGHTS | s_2415_614  |
| Since I believe that I am a sex addict and a sexual anorexic, that means I have a disease of the mind, body, and spirit that reacts very negatively to all sexual stimuli.                                                                                                                                                                          | NO_SYMPTOM     | SPECIAL_CASE      | s_2833_1068 |
| But now I feel like Im hitting a wall, it becomes so frustrating I feel like I get burnt out, and then end up feeling like I need to force myself into doing something however I just end up thinking about it all day and maybe do it last minute if that, and if I dont do it i really beat myself up over it the next day and it becomes harder. | ANHEDONIA      | WORTHLESSNESS     | s_1551_7    |
| I get depressed about the lost youth I once had.                                                                                                                                                                                                                                                                                                    | WORTHLESSNESS  | DEPRESSED_MOOD    | s_485_974   |
| I have a long list of books I'd like to read, but my concentration is terrible.                                                                                                                                                                                                                                                                     | NO_SYMPTOM     | COGNITIVE_ISSUES  | s_2679_133  |
| After the third day, I just couldn't concentrate on anything.                                                                                                                                                                                                                                                                                       | NO_SYMPTOM     | COGNITIVE_ISSUES  | s_1683_76   |
| I can eat whatever I see as optimal for my health, because my sense of reward is not dependent on my dietary choices.                                                                                                                                                                                                                               | NO_SYMPTOM     | APPETITE_CHANGE   | s_1234_37   |
| I'm already happy with who I am and what I do                                                                                                                                                                                                                                                                                                       | NO_SYMPTOM     | SPECIAL_CASE      | s_1404_495  |
| Im so angry now.                                                                                                                                                                                                                                                                                                                                    | NO_SYMPTOM     | DEPRESSED_MOOD    | s_2953_1058 |

## Notes

- PSYCHOMOTOR has only 1 test sample — F1 unreliable for this class
- APPETITE_CHANGE has only 4 test samples — F1 should be interpreted with caution
- NO_SYMPTOM has low recall (model prefers detecting symptoms) — safe bias for screening
- All baselines are from the ReDSM5 paper (CIKM 2025, arXiv:2508.03399)