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

## 2. Target and validation
Claims are matched to a policy row by `policy_id` and a loss date inside `[period_start, period_end]`; net cost = incurred - recoveries, repudiated = 0
(`prep.attach_claims`). Training caps each claim at the line's 99.9th percentile; scoring is always against **uncapped** cost. Validation is by time
(fit on years before Y, score Y, for Y = 2023 and 2024), never random folds: about 85% of 2025 policies appear in training, so random folds would leak.
**One rule is used for every decision in both parts: a change is adopted only if it improves D2 on both holdouts.**

## 3. Data traps (trust and validation)
**A leak.** `customer_service_contacts` raised validation D2 from 0.06 to **0.64**: it correlates 0.75 with same-row claim count, zero contacts means
essentially zero claims, and its 2025 mean is a flat 0.40 against 0.5-0.75 in training. It is an outcome, not a rating factor, and is excluded with
`renewal_decision` (training only); a test enforces it. Noise fields (surname, email domain, campaign, colour, ...) and the 300-level `make_model`
(57% of importance, lower holdout D2) were dropped by ablation (`exp_out.txt`, `exp2_out.txt`).

**Exposure in the wrong unit (now a guardrail).** About 0.4% of training rows carry `exposure` of 365 (days) or 1.014 (a 360-day basis); test rows never do.
The 440-570 rows per line at 365 alone add 160,000-210,000 phantom policy-years (27-29% of a line's exposure), deflating training rates. We found it by
chasing an unexplained gap: pricing features gained +0.02 D2 on the holdouts but only +0.0017 on the leaderboard. The drift screen compared means and
missed it. Exposure is now recomputed from the policy dates and `guardrails.exposure_check` blocks any submission with exposure outside (0, 1] or
inconsistent with the dates. Found on Part 1's last day (too late to submit); the same defect (452 rows) was caught on first load in Part 2.

**Pricing features (v6).** `written_premium`, `discount_applied` and premium per unit exposure are known at inception and pass the leak and drift screens.
On the polluted holdouts they looked strong; on clean data the gain is about +0.002 for buildings and contents and nil for motor (`recal_eval_out.txt`),
and the private leaderboard agrees (v5 0.1265 vs v6 0.1260). They are also circular - they encode the current tariff - which matters in Part 2 (section 6).

## 4. Part 1 model: four validated steps
| Step | What | Holdout evidence (2023 / 2024) | Public LB |
|---|---|---|---|
| Claim development | Per-peril chain-ladder reporting lag; train on exposure x reported fraction (2024: 80% motor, 93% buildings, 97% contents) | 2024 motor cost per exposure 343 reported -> 430 developed, in line with 2022-23 | 0.0859 -> 0.0955 |
| Frequency x severity | LightGBM Poisson x Gamma per line, strong regularisation (15 leaves, 1,000 rows per leaf), 5 seeds | Beat Tweedie pure premium in 6 of 6 line-holdout cases (motor 0.056 / 0.064 vs 0.046 / 0.053) | -> 0.1039 |
| Cost trend | Log-linear frequency + severity trend from pre-holdout years only, damped x0.75 | Buildings and contents were 7-12% under-predicted on both holdouts; trend raised D2 in 4 of 4 cases, A/E 0.98-1.04. Motor trend ~0 and sign-unstable (factor 0.996) | -> 0.1119 |
| Per-peril (buildings, contents) | One frequency x severity pair per peril, summed | Buildings 0.0865 / 0.0937 -> 0.0917 / 0.0945; contents 0.0651 / 0.0710 -> 0.0692 / 0.0725; mixed for motor (not used) | -> 0.1130 |

Caveat: on the immature 2024 holdout, scored against *reported* cost, claim development looks worse (motor 0.052 vs 0.069) because 2024 is almost uniformly
80% reported; the mature 2023 holdout and the leaderboard confirm the gain. Rejected on the same rule: a credibility-weighted vehicle relativity, dropping
weather, looser trees. The trend is the least certain assumption: 2025 is about 2.7 years from the training mean, further than any holdout tested.

## 5. Part 2: what changed in the emerging market (`p2_profile_out.txt`, `p2_segments_out.txt`)
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

## 6. Part 2: how the model adapted
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

**Further ideas, same rule** (`p2_adjust_out.txt`, `p2_peril_*_out.txt`, `p2_pricing_*_out.txt`):

| Idea | Result (D2, time / group holdout) | Decision |
|---|---|---|
| New-business loading (credibility-weighted) | Helps one holdout, hurts the other on every line; the factor itself flips (contents 0.96 vs 1.24) - noise on 37-78 claims | Rejected |
| Developed-market cost trend | Lowers D2 on every line (buildings 0.090 -> 0.081) | Rejected |
| Per-peril models (Part 1's best step) | Contents 0.100 -> 0.096, buildings 0.090 -> 0.069: each peril recalibrated on 32-210 claims (2023 fire x2.0 on 49) | Rejected |
| Pricing features | Motor 0.0890 / 0.0350 -> 0.0892 / 0.0373; buildings 0.0898 / 0.0926 -> 0.0905 / 0.0964; contents 0.0998 -> 0.0991 | Motor and buildings only |

Splitting the data finer (per peril) helped in the data-rich market and hurt in the thin one - the clearest example of the method changing with the market.

**Two submissions, by design** (`p2_final.py`). v1 uses no tariff inputs (public 0.15609); v2 adds the pricing features for motor and buildings (public 0.15611).
v2 predicts 2025 claims slightly better on our holdouts, but it learns partly from the current tariff, which section 7 shows is 2-4x below technical. A repricing
built on v2 would be anchored to the price it is replacing, so **v1 is our recommended model for pricing decisions** and v2 a forecasting benchmark.
v1 predicts 2025 cost per year of exposure of motor 553 (new business 591, renewals 426), buildings 448 and contents 92, against emerging developed history of
444-480, 330-516 and 90-91: motor sits above history because of the younger, new-business-heavy mix; buildings sits between the fire year and the quiet year by design.

## 7. Part 2: office premium for the 20 reference policies (`p2_premium.py`, `outputs/p2_reference_premiums.csv`)
Office premium P = (R + cat + F) / (1 - v - c - t - m), where R is our predicted claim cost, cat = 12% of R for buildings and contents, F = 85 CU per row,
v = 15% variable expense, c = commission by channel (broker 20%, direct 0%), t = 8% premium tax and m = 8% target profit (emerging column of `expenses`).
For broker business the percentage loadings total 51%, so P is about twice R + F. Results: office premium / charged premium averages **2.1 motor, 2.3 contents,
4.1 buildings**; on several policies R alone exceeds the charged premium (e.g. B00217584: R = 1,241 vs 323 charged, flagged for underwriter review).
The acquired broker book is priced well below technical. Assumptions: the emerging comparison-site commission is missing in `expenses` (the developed 10% is used
and flagged; no reference policy uses that channel); the fixed expense is charged per row, not pro-rated, which overstates the ratio for short rows (exposure 0.07, 0.11).

## 8. AI-assisted workflow and human review (effective AI use)
| Stage | AI contribution | Human review / verification |
|---|---|---|
| Scoping | Read the rules, derived the target and metric, proposed the pipeline | Team confirmed objective and constraints |
| Leak and data errors | Profiling, contacts-vs-claims, train/test checks; traced the leaderboard-vs-holdout gap to exposure in days | Reviewed evidence; approved exclusions and the exposure fix; regression tests added |
| Experiments | Ablations, model comparisons, holdout designs with fixed seeds and cached frames | Every decision is backed by a result file in the repo |
| Claim development | Proposed chain-ladder reporting lag; caught its own bug (new column leaking in as a feature) | Checked curves against raw lag percentiles; challenged the validation artefact |
| Part 2 transfer | Profiled the new market, harmonised fields, designed four variants, two holdouts and four follow-up tests | Chose the worst-case selection rule; decided to submit v2 but recommend v1 because of tariff circularity |
| Guardrails and tests | Wrote `guardrails.py` and 31 pytest checks | AI-written tests that were wrong were fixed on evidence, not loosened: a drift test on the leak column, an expectation that a 360-day row breaks the date check (it is within tolerance; the >1 check catches it), and a column-order check (columns are used by name) |
| Operations | Kaggle CLI behind the corporate proxy (Windows certificate store); diagnosed a full C: drive as the cause of memory errors | Team approved joining Part 2, supplying the API token and deleting files to free space |
| Reporting | Drafted this document from the result files | Numbers checked against the output files |

AI was not trusted on its own claims: three of its own hypotheses (pricing features in Part 1; the new-business loading and per-peril models in Part 2) looked convincing and were cut back or rejected when validated.

## 9. Governance and guardrails
`final_v3.py` and `p2_final.py` refuse to write a submission unless: exposure is within (0, 1] (`exposure_check`); no feature is flagged by the leak screen
(|correlation| > 0.30 with same-row claim count); the file has exactly the sample row ids, unique, finite and non-negative. Each line's 2025 cost per exposure is
checked against a +/-25% band around developed history (a Part 2 failure prints a human-review warning rather than silently passing). Drift flags and every
calibration factor are written to `outputs/report_*.json`. 31 automated tests (22 Part 1, 9 Part 2) cover forbidden columns, leak / drift / exposure screens,
time-split integrity, the reporting curve, market harmonisation and shared vocabulary, office-premium arithmetic and submission validity.

## 10. Limitations and remaining uncertainty
1. **Emerging buildings level**: two years, fire-dominated (70 claims = 45% of cost). Our level leans on the developed market; a single large fire in 2025 would dominate the result.
2. **New business**: 76% of the 2025 book. The new-business effect comes mostly from developed-market structure; the emerging evidence is too thin to confirm or correct it.
3. **Trend**: no trend in Part 2 (the developed trend failed validation). If emerging claims inflation exceeds developed, predictions are low. Part 1's trend is a four-point extrapolation damped by judgement.
4. **Weather and severity**: 2025 emerging weather is partly outside the training range (drift flagged); emerging severity structure is borrowed entirely from the developed market.
5. **Tariff circularity (v2)**: the pricing features encode the current, under-priced tariff; use v1 for repricing.
6. **Tail, development and selection**: no extreme-value treatment; case-reserve development of open claims is not modelled; several candidates per decision leave some selection optimism despite the both-holdouts rule.

## 11. Reproducibility
`pip install -r requirements.txt`; Part 1 data in `data/public/`, Part 2 data in `data/part2/{developed,emerging}/`; then follow the command list in `README.md`.
All experiments log to `*_out.txt` / `*_results.csv` files in the repo; seeds are fixed (bagging seeds 0-4).
