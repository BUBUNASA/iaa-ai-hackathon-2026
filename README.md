# IAA AI-Assisted Hackathon 2026: claim-cost (risk-premium) models for a developed and an emerging market

Predicts the expected ultimate claim cost of every 2025 motor, buildings and contents policy row (metric: Tweedie D2, power 1.5).

**Part 1 - developed market** (final, private leaderboard = 67% of test): v1 0.1025 -> v2 claim development 0.1101 -> v3 freq x sev 0.1160 ->
v4 cost trend 0.1226 -> v5 per-peril 0.1265 -> v6 pricing features 0.1260 (selected; rank 43). Public scores: 0.0859 / 0.0955 / 0.1039 / 0.1119 / 0.1130 / 0.1147.

**Part 2 - emerging market**: p2 v1 (developed-market structure transferred and recalibrated) public 0.15609 (rank 13 / 18 at submission);
p2 v2 (+ current-tariff pricing features for motor and buildings) public 0.15611. See `REPORT.md`.

## Part 1 (developed market) approach
| Step | What | Why (evidence) |
|---|---|---|
| Target | Claims matched to policy rows by `policy_id` + loss date in period; net of recoveries, repudiated = 0; per-claim cap at the 99.9th percentile for training | Heavy tail: top 0.1% of rows hold 20-32% of cost |
| Leak removal | `customer_service_contacts` and `renewal_decision` excluded | Contacts track the same-row claim count (corr 0.75; D2 0.64 vs 0.06 without) and are flat at 0.40 in 2025 test vs 0.5-0.75 in training |
| Claim development | Chain-ladder reporting-lag CDF per peril; training exposure x reported fraction | 2024 claims valued at 31-Dec-2024 are immature (motor ~80% reported); developed 2024 motor cost/exposure 430 vs 343 raw. LB +0.0096 |
| Model | LightGBM Poisson frequency x Gamma severity, strong regularisation, 5-seed bagging | Beat 3-seed Tweedie pure-premium in 6/6 line x holdout cases (buildings 2023 only marginally; see `blend_eval_out.txt`). Seed bagging helped Tweedie in 4/6 cases (not yet measured for freq x sev) |
| Governance | Automated leak, drift, format and sanity guardrails (`guardrails.py`, `tests/`) | Run before any submission is written |
| Per-peril models | Buildings and contents: one freq x sev pair per peril, each on its own developed exposure, summed (`perils.py`) | Beat the line model on both holdouts (buildings 0.0865/0.0937 -> 0.0917/0.0945; contents 0.0651/0.0710 -> 0.0692/0.0725). Mixed for motor, so not used there (`improve_eval_out.txt`) |
| Cost trend | Freq + severity log-linear trend from train years only (`trend.py`), damped x0.75, applied to all lines | Buildings/contents were 7-12% under-predicted on both holdouts. A trend estimated on pre-holdout years only improves D2 in 4/4 buildings/contents cases (A/E 0.98-1.04 at x0.75). Motor trend is ~0 and sign-unstable, so its 2025 factor is ~0.996 (`trend_eval_out.txt`) |

Rejected ideas (all validated, see `*_out.txt`): vehicle `make_model` (300 levels; overfits), weather/history features (tiny, noisy),
`written_premium`/`discount_applied` (risk under market change), an ad-hoc multiplicative level scalar (motor is calibrated, A/E 0.99-1.00; buildings/contents are 7-12% under-predicted on both holdouts but that tracks cost trend, not a cap/level artefact - see Open issues).

Data-quality fix (found after v6): ~0.4% of training rows record `exposure` in days (365) or on a 360-day basis (1.014); test rows never do.
`prep.load_policies` recomputes them from the policy dates and `guardrails.exposure_check` now blocks any submission built on exposure > 1.
The same defect is present in the emerging market. Re-validation on clean data: `recal_eval_out.txt`, `pricing_diag_fixed_out.txt`.

## Part 2 (emerging market) approach
| Step | What | Evidence |
|---|---|---|
| Profile | Emerging vs developed: schema, data quality, drift, experience by year / region / peril / segment (`p2_profile.py`, `p2_segments.py`) | 2025 test is 76% new business (7.6% of training exposure); motor claims 1.3x as frequent but 0.63x as costly as developed |
| Harmonise | Age bands, cleaned free-text occupation + self-employed flag, construction codes mapped, developed-only fields dropped, gender kept, market flag (`p2_features.py`) | Identical columns in both markets (asserted) |
| Transfer | Four variants (`p2_models.py`): emerging-only; developed model recalibrated (freq and sev separately); pooled with market flag; developed model as an offset boosted on emerging data | Borrowing from developed beats emerging-only on every line, both holdouts (bootstrap P > 0.9) |
| Validate | Two holdouts: emerging 2024 (time) and 30% of emerging policies (group); new-business subset and capped-claim views (`p2_eval.py`, `p2_blend.py`) | Equal blend of the 3 transfer variants is within 0.001 of the best worst-case D2 on every line |
| Price | Office-premium build-up for the 20 reference policies from the `expenses` table (`p2_premium.py`) | Technical premium is 2-4x the charged premium: the acquired book is under-priced |

## Reproduce
```
pip install -r requirements.txt
# put the Kaggle data in data/public/*.parquet
python exp.py            # builds cached feature frames (data/cache)
python val_preds.py      # validation predictions (2023 & 2024 holdouts)
python blend_eval.py     # model / blend / load decisions
python final_v3.py --seeds 5 --w-fs 1.0 --trend 0.75 --peril-lines buildings contents --tag peril   # final: outputs/submission_peril.csv (guardrails run inside)
python -m pytest -q      # guardrail tests

# Part 2: Kaggle part-2 data unzipped into data/part2/{developed,emerging}/
python p2_profile.py > p2_profile_out.txt        # what changed between markets
python p2_features.py                            # harmonised, cached frames (data/cache_p2)
python p2_eval.py --scheme time; python p2_eval.py --scheme group; python p2_blend.py   # variant choice
python p2_final.py --blend DEV_CAL,POOL,OFFSET --seeds 5 --tag p2                     # outputs/submission_p2.csv
python p2_premium.py outputs/submission_p2.csv  # office premium build-up for the 20 reference policies
```

## Files
`prep.py` target construction | `features.py` features and exclusions | `dev.py` claim development | `models.py` model zoo |
`guardrails.py` + `tests/` governance | `final_v3.py` final pipeline | `eda_profile.py`, `leakage_check.py`, `dev_explore.py`, `exp*.py` analysis |
Part 2: `p2_features.py` harmonisation | `p2_models.py` transfer variants | `p2_eval.py`, `p2_blend.py`, `p2_adjust.py` validation | `p2_final.py` submission | `p2_premium.py` office premium
