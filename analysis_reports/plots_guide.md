# Study Plots – Quick Guide

This note explains, in plain language, what each generated plot shows and how to read it. Higher = better means more stable or more agreement, unless stated otherwise.

## A. Model-Level Consistency (runs 1–4, temp=1.0)

- A1: `A1_model_yesno_perfect_consistency.png`
  - What: For each model, the percentage of (question, language) items where all 4 Yes/No answers were identical.
  - Read it: Higher bars = more repeatable Yes/No answers across runs.

- A2: `A2_model_yesno_avg_flip_rate.png`
  - What: For each model, how often the Yes/No answer changed compared to run 1 (averaged over items and runs 2–4).
  - Read it: Lower bars are better (fewer flips from the first run).

- A3: `A3_model_scale_high_consistency.png`
  - What: For each model, the percentage of scale answers (1–10) whose standard deviation across runs is ≤ 1.
  - Read it: Higher bars = scale scores are tightly clustered across runs.

- A4: `A4_model_scale_avg_std.png`
  - What: Average standard deviation of scale answers across runs.
  - Read it: Lower bars are better (less variability).

## B. Temperature Effect (runs 1–4 vs run 5)

- B1: `B1_temp_yes_agree_by_model.png`
  - What: Agreement rate of Yes/No answers when switching from temp=1.0 (mean of runs 1–4) to temp=0.6 (run 5), per model.
  - Read it: Higher bars = model keeps the same Yes/No at lower temperature.

- B2: `B2_temp_yes_agree_by_language.png`
  - What: Same agreement rate as B1, but averaged per language.
  - Read it: Which languages are more/less sensitive to temperature changes.

- B3: `B3_temp_scale_mad_by_model.png`
  - What: Mean absolute difference of scale scores between mean(temp=1.0) and temp=0.6, per model.
  - Read it: Lower bars are better (scale scores change less with temperature).

- B4: `B4_temp_scale_mad_by_language.png`
  - What: Same as B3, but averaged per language.
  - Read it: Which languages show bigger/smaller shifts in scale scores.

## C. Language-Level Consistency (runs 1–4)

- C1: `C1_lang_yesno_perfect_consistency.png`
  - What: For each language, the percentage of items with identical Yes/No across runs.
  - Read it: Higher is better; the language is more stable.

- C2: `C2_lang_yesno_flip_rate.png`
  - What: Average flip rate vs run 1 for Yes/No, per language.
  - Read it: Lower is better (fewer run-to-run changes).

- C3: `C3_lang_scale_high_consistency.png`
  - What: Percentage of scale answers with std dev ≤ 1, per language.
  - Read it: Higher is better (stable scale answers).

- C4: `C4_lang_scale_avg_std.png`
  - What: Average std dev of scale answers, per language.
  - Read it: Lower is better (less variability).

## D. Cross-Model & Cross-Language Agreement (runs 1–4)

- D1: `D1_pairwise_yes_agreement.png`
  - What: Heatmap of model × model agreement on Yes/No (fraction of identical modal answers).
  - Read it: Brighter cells (closer to 1.0) mean two models answer Yes/No similarly.

- D2: `D2_pairwise_scale_correlation.png`
  - What: Heatmap of model × model correlation on scale scores (means across runs).
  - Read it: Positive correlation = models give similar numeric ratings across items.

- D3: `D3_lang_agreement_per_question.png`
  - What: Heatmap of how much each language agrees with the question-level majority (across models), per question.
  - Read it: Higher = that language often matches the overall consensus on that question.

## E. Question-Level Reliability (runs 1–4)

- E1: `E1_question_model_yesno_stability.png`
  - What: Heatmap of Yes/No stability per question × model (1 = identical across runs, 0 = changed at least once), averaged over languages.
  - Read it: Brighter cells = the model is stable on that question.

- E2: `E2_boxplot_scale_std_per_question.png`
  - What: For each question, distribution of scale standard deviations across models × languages.
  - Read it: Lower medians and tighter boxes = more reliable numeric scoring for that question.

## F. Temperature Sensitivity Deep Dive

- F1: `F1_hist_scale_diffs_per_model.png`
  - What: For each model, histogram of |Temp0.6 − mean(Temp1.0)| on scale items.
  - Read it: Histograms concentrated near 0 indicate minimal temperature impact.

- F2: `F2_scatter_scale_temp_comparison_per_model.png`
  - What: For each model, scatter of Temp0.6 (y) vs mean Temp1.0 (x) with a trendline.
  - Read it: Points near the diagonal and a slope ≈ 1 indicate stable behavior.

## G. Composite Ranking of Models

- G1: `G1_radar_<model>.png`
  - What: Radar (spider) chart per model, with four axes: Yes/No consistency, 1 − flip rate, scale stability, temperature agreement.
  - Read it: Larger filled area = stronger overall stability.

- G2: `G2_composite_stability_index.png`
  - What: Bar chart of a single “Stability Index” combining the four axes above (normalized to 0–1 and averaged).
  - Read it: Higher = more stable and consistent overall.

---

Notes
- Bars are sorted to make comparisons easy.
- Heatmaps include numeric annotations where helpful; closer to 1.0 means higher agreement/stability.
- Temperature comparisons always use mean of runs 1–4 (temp=1.0) vs run 5 (temp=0.6).
