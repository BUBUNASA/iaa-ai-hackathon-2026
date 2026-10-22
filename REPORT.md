# Risk-premium modelling for a motor / buildings / contents book: leak-free, claim-developed, governed

**Team:** _[team name / members]_  **Competition:** IAA AI-Assisted Hackathon 2026, Part 1  **Repo:** this repository (`README.md` to reproduce)
**Public leaderboard (Tweedie D2, higher is better):** v1 baseline 0.0859 -> v2 + claim development 0.0955 -> v3 freq x sev + bagging 0.1039 -> v4 + cost trend 0.1119 -> **v5 final (per-peril buildings/contents) 0.1130**

## 1. Problem and approach in one paragraph
For each 2025 policy row we predict expected ultimate claim cost (net of excess and recoveries, repudiated excluded), i.e. the risk premium
(626,639 rows, three lines). Three data facts drive every design choice: (i) 90-97% of rows have zero cost and the top 0.1% of rows hold 20-32% of
cost; (ii) one customer field leaks the outcome; (iii) 2024 claims are valued at 31-Dec-2024 and are still immature.
The final model is a LightGBM **Poisson frequency x Gamma severity** model per line, trained on **developed exposure**, 5-seed bagged, behind
automated **leak / drift / format / sanity guardrails**.

## 2. Target construction and validation design
* Claims are matched to a policy row by `policy_id` and a loss date inside `[period_start, period_end]`; net incurred = incurred - recoveries, repudiated = 0.
  Duplicate matches and unmatched claims are counted (`prep.attach_claims`).
* For training only, each claim is capped at the 99.9th percentile of its line to stabilise the severity model. Scoring is always against the **uncapped** cost.
* **Time-based validation**: fit on years before Y, score on Y, for Y = 2023 and 2024. Random k-fold would leak (the same policy appears in several years;
  about 85% of 2025 policies were seen in training), so it is not used. Two holdouts are used because 2024 is immature (section 4).

## 3. Finding and removing a leak (trust and validation)
`customer_service_contacts` raised validation D2 from about 0.06 to **0.64**. Evidence it is a same-period outcome proxy, not a pricing factor:
correlation 0.75 with claim count; zero contacts implies essentially zero claims; and its 2025 mean is 0.40 against 0.5-0.75 in training years.
It is excluded, as is `renewal_decision` (train only). A model using it would have scored brilliantly in validation and been wrong in production.
This is enforced by `tests/test_guardrails.py` (the leak screen must flag the known leak, and no feature may correlate above 0.30 with same-row claim count).

Other exclusions, each by ablation (`exp_out.txt`, `exp2_out.txt`): noise columns (surname, email domain, campaign code, branch, colour, ...) that cost D2 when included;
the 300-level vehicle `make_model` (overfits: 57% of importance, lower holdout D2); `written_premium` / `discount_applied` (small gain, but circular and fragile under a market change).

## 4. Claim development (largest single gain: +0.0096 on the leaderboard)
Motor 2024 cost per exposure was 343 against 426-441 in earlier years. The cause is reporting lag, not better risk: median report lag is 9 days, but bodily-injury claims average 181 days and run to about 1,400.
We estimate a per-peril reporting-lag distribution with a chain-ladder on the (loss-age x lag) claim-count triangle, weight perils by cost, and convert it into a reported fraction per
policy row (about 80% motor, 93% buildings, 97% contents for 2024). Training exposure is multiplied by this fraction so the model learns **ultimate** rates, which is what 2025 requires.
Developed 2024 motor cost per exposure becomes 430, consistent with 2022-23.

*Validation caveat (important for reviewers).* On the 2024 holdout, scored against *reported* cost, the developed model looks worse (motor D2 0.052 vs 0.069).
Almost all 2024 exposure starts in January, so the reported fraction is a near-constant 0.80 and the comparison merely penalises predicting ultimate cost.
The more mature 2023 holdout shows no such penalty, and the leaderboard (which pays on 2025 ultimate cost) confirms the benefit. We disclose this rather than cherry-pick the favourable view.

## 5. Model choice and calibration
On both holdouts, scored on the reported basis, frequency x severity beat Tweedie pure-premium in every line-holdout case (D2: motor 0.0555 / 0.0640 vs 0.0463 / 0.0531 for 2023 / 2024;
buildings 0.0785 / 0.0850 vs 0.0782 / 0.0797; contents 0.0620 / 0.0635 vs 0.0565 / 0.0579 - `blend_eval_out.txt`).
Blends with Tweedie were intermediate, so the final model is pure frequency x severity (a 3-seed Tweedie average was the comparator).
Strong regularisation (15 leaves, 1000 minimum rows per leaf, L2 = 100, categorical smoothing) keeps the model stable on a 3-4% claim frequency.

Calibration (actual / predicted, reported basis): motor 0.99-1.00 on both holdouts. Buildings 1.08 / 1.12 and contents 1.07 / 1.08, i.e. these lines are **under-predicted by 7-12%**
on both holdouts. This is not a cap artefact (the cap load is only about 1.00-1.01). Developed buildings cost per exposure rose 283 -> 309 -> 322 -> 367 over 2021-24, so the likeliest
cause is unmodelled cost trend. We deliberately did **not** apply a level scalar tuned to the leaderboard. Instead we **validated a cost-trend factor on the holdouts first** (`trend_eval.py`): a log-linear
frequency and severity trend estimated on years before the holdout only. It raised D2 in all four buildings/contents holdout cases (buildings 0.0785 -> 0.0865 and 0.0850 -> 0.0937; contents 0.0620 -> 0.0651 and 0.0635 -> 0.0710 at 0.75 damping)
and brought A/E to 0.98-1.04. Motor's trend is near zero and changes sign between holdouts (a full trend would have hurt the 2024 holdout), so its 2025 factor is 0.996, i.e. effectively none.
We damp the trend by 0.75 because full trend slightly over-corrected buildings (A/E 0.95) and half under-corrected it (1.05). The 2025 factors are buildings 1.163 and contents 1.105. Public leaderboard: 0.1039 -> 0.1119.
Caveat: 2025 is further from the training mean (about 2.7 years) than the holdouts were (1.5-2 years), so the extrapolation is the least certain assumption in the model.

**Per-peril models (buildings, contents).** Perils have different drivers (storm/flood follow weather and location, escape of water follows property age, theft follows region) and different reporting lags.
We fit a frequency x severity pair per peril, each on its own developed exposure, and sum them. Four candidates were judged on both holdouts with the same trend basis (`improve_eval_out.txt`); only this one beat the line model on both years:

| Line | Line model D2 (2023 / 2024) | Per-peril D2 (2023 / 2024) | Adopted |
|---|---|---|---|
| Buildings | 0.0865 / 0.0937 | 0.0917 / 0.0945 | yes |
| Contents | 0.0651 / 0.0710 | 0.0692 / 0.0725 | yes |
| Motor | 0.0608 / 0.0598 | 0.0633 / 0.0576 | no (mixed) |

Rejected on the same test: a credibility-weighted vehicle code relativity (worse on both years), dropping weather features and looser trees (mixed). Public leaderboard: 0.1119 -> 0.1130.

## 6. AI-assisted workflow and human review (effective AI use)
| Stage | AI contribution | Human review / verification |
|---|---|---|
| Scoping | Read the competition rules, derived the target and metric, proposed the pipeline | Team confirmed the objective and constraints |
| EDA and leak detection | Generated profiling and the contacts-vs-claims, train/test distribution and ablation checks | Reviewed the numbers; decided on exclusion and added a regression test so it cannot return |
| Feature / model experiments | Wrote and ran ablations, regularisation and model-zoo comparisons with fixed seeds and cached frames | Each decision is backed by a printed result file in the repo |
| Claim development | Proposed chain-ladder reporting lag; AI-caught bug: the new column was entering the model as a feature (fixed and re-run) | Checked curves against raw lag percentiles; challenged the validation artefact |
| Guardrails and tests | Wrote `guardrails.py` and 20 pytest checks | One AI-written test was wrong (drift on the leak column was weaker than weather drift); we removed it rather than loosen thresholds to pass |
| Reporting | Drafted this document from the result files | Numbers checked against the output files |

## 7. Governance and guardrails
`final_v3.py` refuses to write a submission unless: no feature is flagged by the leak screen; the submission has exactly the sample row ids, unique, finite, non-negative;
and predicted 2025 cost per exposure lies within +/-25% of the developed historical range for each line. Drift flags are written to `outputs/report_final.json`
(none above threshold in the final run). 20 automated tests cover forbidden columns, leak and drift screens, time-split integrity, the reporting curve (valid, monotone CDF), and submission validity.
Final 2025 predicted cost per exposure: motor 433, buildings 397, contents 98.6 (developed history: motor 417-442, buildings 283-367, contents 76-88). Buildings and contents sit above the historical range by design (trend); the sanity band still passes. Per-peril buildings over-predicted the 2024 holdout slightly (A/E 0.94), so buildings is the line most at risk of being too high.

## 8. Limitations and next steps
1. **Trend**: the factor is a four-point extrapolation (4 years of history); the damping of 0.75 is a judgement. Consider external inflation indices in Week 2.
2. **Tail**: no extreme-value treatment; the largest motor row is 11.9M. The cap protects training but 2025 large losses remain irreducible noise for D2.
3. **Development tail**: reporting lag is modelled, case-reserve development of open claims is not (open claims average 6k incurred vs 1.2k paid for 2024 motor).
4. **Public vs final score**: the public leaderboard uses part of the test set; we selected on the cross-year holdouts plus the leaderboard, so some optimism is likely.
5. **Week 2 (new market)**: reuse the pipeline; expect shifted exposure mix, so run the drift screen first, re-fit severity and frequency on the new market's own history if available, and keep `written_premium` excluded.

## 9. Reproducibility
`pip install -r requirements.txt`, place the Kaggle parquet files in `data/public/`, then follow the command list in `README.md`. All experiments log to `*_out.txt` files in the repo; seeds are fixed (bagging seeds 0-4).
