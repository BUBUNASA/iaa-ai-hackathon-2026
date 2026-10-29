# Risk-premium models for a developed and an emerging market: leak-free, claim-developed, transferred with evidence

**Team:** NazaliBupa  **Competition:** IAA AI-Assisted Hackathon 2026, Parts 1 and 2  **Repo:** this repository (`README.md` to reproduce)

| Leaderboard (Tweedie D2, higher is better) | v1 baseline | v2 claim development | v3 freq x sev | v4 cost trend | v5 per-peril | v6 pricing features |
|---|---|---|---|---|---|---|
| Part 1 public | 0.0859 | 0.0955 | 0.1039 | 0.1119 | 0.1130 | **0.1147** |
| Part 1 private (final, 67% of test) | 0.1025 | 0.1101 | 0.1160 | 0.1226 | **0.1265** | 0.1260 (selected; rank 43) |

Part 2 (emerging market): developed-market structure transferred and recalibrated, public **0.1561** (v1, no tariff inputs; rank 13 of 18 at submission);
v2 adds the current tariff (written premium, discount, premium rate) for motor and buildings only, public **0.1561** (0.15611 vs 0.15609).

## 1. Problem and approach in one paragraph
For each 2025 policy row we predict expected ultimate claim cost (net of excess and recoveries, repudiated excluded), i.e. the risk premium,
for motor, buildings and contents. Part 1 is an established market (4 years, 626,639 test rows); Part 2 an emerging market the insurer entered
2 years ago and is now growing through a broker portfolio (209,063 test rows, 76% new business). Data facts drive every design choice:
(i) 90-97% of rows have zero cost and the top 0.1% of rows hold 20-32% of cost; (ii) one customer field leaks the outcome; (iii) claims are valued
at 31-Dec-2024 so recent years are immature; (iv) about 0.4% of rows record exposure in the wrong unit. The model is a LightGBM
**Poisson frequency x Gamma severity** pair per line, trained on **developed exposure**, 5-seed bagged, behind automated guardrails.
In Part 2 the emerging data is too thin to learn structure on its own, so the developed-market model supplies it and the emerging data recalibrates it.

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
the 300-level vehicle `make_model` (overfits: 57% of importance, lower holdout D2).

**Exposure recorded in the wrong unit (found late; a guardrail now).** About 0.4% of training rows in every year and line carry `exposure` of 365
(days) or 1.014 (a 360-day basis); test rows never do. The 440-570 rows per line at 365 alone add 160,000-210,000 phantom policy-years (27-29% of a line's exposure),
deflating training rates and distorting holdout scores. We found it by chasing an unexplained gap: pricing features gained +0.02 D2 on the holdouts but only
+0.0017 on the leaderboard, and the pricing model sat 24-34% above the base model on the test set while sitting below it on the holdouts. The drift screen
compared means only and missed it. Exposure is now recomputed from the policy dates (`prep.load_policies`), and `guardrails.exposure_check` blocks any
submission with exposure outside (0, 1] or inconsistent with the dates. The fix came on the last day of Part 1 and was not submitted in time;
the same defect is present in the emerging market (452 rows) and was caught on first load in Part 2.

**Pricing features (v6), and what the private leaderboard said.** `written_premium`, `discount_applied` and premium per unit exposure are known at inception,
passed the leak and drift screens, and looked strong on the (polluted) holdouts. Because premium inflates ~3-5% a year they also shifted the level, so v6 rescaled
the pricing model to the base model's total. On clean data the gain is small: +0.002 D2 for buildings and contents on both holdouts, none for motor, and the
rescaling no longer helps (`recal_eval_out.txt`). The private leaderboard agrees: v5 0.1265 vs v6 0.1260. Pricing features are also circular (they encode the
current tariff), so they were **not** carried into Part 2, where the tariff is the thing under review.

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

## 6. Part 2: what changed in the emerging market (`p2_profile_out.txt`, `p2_segments_out.txt`)
| Dimension | Developed (Part 1) | Emerging (Part 2) | Consequence |
|---|---|---|---|
| History | 2021-24, ~163,000 claims | 2023-24, 9,258 claims (18x fewer) | Emerging data alone cannot learn structure: an emerging-only severity model stops after one tree |
| 2025 book | 15% new business | **76% new business** (7.6% of training exposure); 24% of test policies seen before; drivers/policyholders aged 17-34: 50% vs 39% | The model must price customers it has no history for |
| Motor | 0.13-0.15 claims per year, ~3,800 per claim | 0.18-0.19 claims per year, ~2,200 per claim | Recalibrate frequency (x1.34) and severity (x0.63) separately, not just the total |
| Buildings | Cost per year 432-449 (2023-24) | 514 then 307: fire is 45% of cost from 70 claims | Emerging level is volatile; lean on the developed level |
| Contents | Cost per year ~117 (2023-24) | 87-91 | Mostly a level shift |
| Fields | Exact ages, credit band, mileage, telematics | Age bands only; free-text occupation (80 spellings of 20 jobs); new construction codes; `gender` available | Harmonise to the common subset (below) |
| Climate | | Wetter, warmer, 2-3x the hail days; 2025 weather (supplied) milder but with more freezing days | Weather enters as features; 2025 is partly outside the emerging range (drift flagged) |
| Same defects | | Exposure in days (452 rows); `customer_service_contacts` flat at 0.40 in test | Part 1 fixes and guardrails reused unchanged |

Reporting lags are the same in both markets (median 6-9 days, 90th percentile 22-41), so the developed claim-development curves are reused.

## 7. Part 2: how the model adapted
**Harmonisation** (`p2_features.py`, tested in `tests/test_p2.py`): developed ages banded like the emerging bands; developed-only fields dropped; occupation trimmed and
lower-cased into the 20 base jobs plus a `self_employed` flag; construction codes mapped to one vocabulary (Brick/BRK -> brick; W/T/WOD -> timber, matched on
shares of 12.1% vs 12.0%; non_standard/MIX; stone/STN); `gender` kept (permitted in the emerging market) and missing for developed; a `market` flag added.

**Four ways to use the developed market** (`p2_models.py`), all frequency x severity on developed exposure:
EM = emerging data only; DEV_CAL = developed model applied to emerging rows, recalibrated by emerging actual/expected separately for claim count and claim size;
POOL = one model on both markets with the market flag; OFFSET = DEV_CAL as a fixed starting point that boosting on emerging data corrects only where the evidence
supports it (a boosting analogue of credibility).

**Two holdouts** (`p2_eval.py`): *time* (developed <= 2023 + emerging 2023 -> score emerging 2024) and *group* (hold out 30% of emerging policies, all years).
Both are reported in total, on new business only, and against capped claims; a 300-draw bootstrap gives P(beats EM).

| Line | Holdout | EM | DEV_CAL | POOL | OFFSET | Equal blend (chosen) |
|---|---|---|---|---|---|---|
| Contents | time / group | 0.064 / 0.071 | 0.100 / 0.120 | 0.099 / 0.116 | 0.098 / 0.120 | 0.100 / 0.119 |
| Buildings | time / group | 0.031 / 0.032 | 0.080 / 0.096 | 0.097 / 0.079 | 0.082 / 0.099 | 0.090 / 0.093 |
| Motor | time / group | 0.078 / 0.020 | 0.087 / 0.034 | 0.089 / 0.033 | 0.088 / 0.035 | 0.089 / 0.035 |

Borrowing from the developed market beats emerging-only on every line and both holdouts (bootstrap P 0.87-1.00). No single variant wins both holdouts on
every line, so we chose by the **best worst-case D2**; the equal blend of DEV_CAL, POOL and OFFSET is within 0.001 of the best on every line and avoids picking a
different winner per line on noisy evidence. (Motor's low group-holdout D2 is one 136,767 new-business claim, 21% of new-business cost in that holdout.)

**Adjustments tested and rejected** (`p2_adjust_out.txt`). Both holdouts suggested contents new business was under-predicted (actual/expected 1.2-1.4).
A credibility-weighted new-business factor, estimated on the training rows only, helped one holdout and hurt the other on every line, and the factor itself
flipped between holdouts (contents 0.96 vs 1.24), so the apparent gap is noise on 37-78 claims. A cost trend taken from the developed market lowered D2 on the
time holdout for every line (buildings 0.090 -> 0.081). Neither was applied.

**Submission** (`p2_final.py`): predicted 2025 cost per year of exposure motor 553 (new business 591, renewals 426), buildings 448, contents 92, against emerging
developed history of 444-480, 330-516 and 90-91. Motor sits above history because of the younger, new-business-heavy mix; buildings sits between the fire year and
the quiet year by design.

## 8. Part 2: office premium for the 20 reference policies (`p2_premium.py`, `outputs/p2_reference_premiums.csv`)
Office premium P = (R + cat + F) / (1 - v - c - t - m), where R is our predicted claim cost, cat = 12% of R for buildings and contents, F = 85 CU per row,
v = 15% variable expense, c = commission by channel (broker 20%, direct 0%), t = 8% premium tax and m = 8% target profit (emerging column of `expenses`).
For broker business the percentage loadings total 51%, so P is about twice R + F. Results: office premium / charged premium averages **2.1 motor, 2.3 contents,
4.1 buildings**; on several policies R alone exceeds the charged premium (e.g. B00217584: R = 1,241 vs 323 charged, flagged for underwriter review).
The acquired broker book is priced well below technical. Assumptions: the emerging comparison-site commission is missing in `expenses` (the developed 10% is used
and flagged; no reference policy uses that channel); the fixed expense is charged per row, not pro-rated, which overstates the ratio for short rows (exposure 0.07, 0.11).

## 9. AI-assisted workflow and human review (effective AI use)
| Stage | AI contribution | Human review / verification |
|---|---|---|
| Scoping | Read the rules, derived the target and metric, proposed the pipeline | Team confirmed objective and constraints |
| Leak and data errors | Profiling, contacts-vs-claims, train/test checks; traced the leaderboard-vs-holdout gap to exposure in days | Reviewed evidence; approved exclusions and the exposure fix; regression tests added |
| Experiments | Ablations, model comparisons, holdout designs with fixed seeds and cached frames | Every decision is backed by a result file in the repo |
| Claim development | Proposed chain-ladder reporting lag; caught its own bug (new column leaking in as a feature) | Checked curves against raw lag percentiles; challenged the validation artefact |
| Part 2 transfer | Profiled the new market, harmonised fields, designed four variants and two holdouts | Chose the worst-case selection rule; accepted the rejection of the new-business factor and trend |
| Guardrails and tests | Wrote `guardrails.py` and 31 pytest checks | AI-written tests that were wrong were fixed on evidence, not loosened: a drift test on the leak column, an expectation that a 360-day row breaks the date check (it is within tolerance; the >1 check catches it), and a column-order check (columns are used by name) |
| Operations | Kaggle CLI behind the corporate proxy (Windows certificate store); diagnosed a full C: drive as the cause of memory errors | Team approved joining Part 2, supplying the API token and deleting files to free space |
| Reporting | Drafted this document from the result files | Numbers checked against the output files |

AI was not trusted on its own claims: two of its own hypotheses (pricing features, new-business loading) looked convincing and were cut back or rejected when validated.

## 10. Governance and guardrails
`final_v3.py` and `p2_final.py` refuse to write a submission unless: exposure is within (0, 1] (`exposure_check`); no feature is flagged by the leak screen
(|correlation| > 0.30 with same-row claim count); the file has exactly the sample row ids, unique, finite and non-negative. Each line's 2025 cost per exposure is
checked against a +/-25% band around developed history (a Part 2 failure prints a human-review warning rather than silently passing). Drift flags and every
calibration factor are written to `outputs/report_*.json`. 31 automated tests (22 Part 1, 9 Part 2) cover forbidden columns, leak / drift / exposure screens,
time-split integrity, the reporting curve, market harmonisation and shared vocabulary, office-premium arithmetic and submission validity.

## 11. Limitations and remaining uncertainty
1. **Emerging buildings level**: two years, fire-dominated (70 claims = 45% of cost). Our level leans on the developed market; a single large fire in 2025 would dominate the result.
2. **New business**: 76% of the 2025 book. The new-business effect comes mostly from developed-market structure; the emerging evidence is too thin to confirm or correct it.
3. **Trend**: no trend in Part 2 (the developed trend failed validation). If emerging claims inflation exceeds developed, predictions are low. Part 1's trend is a four-point extrapolation damped by judgement.
4. **Weather and severity**: 2025 emerging weather is partly outside the training range (drift flagged); emerging severity structure is borrowed entirely from the developed market.
5. **Tail and development**: no extreme-value treatment; reporting lag is modelled but case-reserve development of open claims is not.
6. **Selection**: decisions were taken on holdouts, not the leaderboard, but with several candidates per decision some optimism remains.

## 12. Reproducibility
`pip install -r requirements.txt`; Part 1 data in `data/public/`, Part 2 data in `data/part2/{developed,emerging}/`; then follow the command list in `README.md`.
All experiments log to `*_out.txt` / `*_results.csv` files in the repo; seeds are fixed (bagging seeds 0-4).
