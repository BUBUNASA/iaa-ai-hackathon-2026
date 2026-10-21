# IAA AI-Assisted Hackathon 2026 - Part 1: claim-cost (risk-premium) model

Predicts the expected ultimate claim cost of every 2025 motor, buildings and contents policy row
(metric: Tweedie D2, power 1.5). Public leaderboard: v1 0.0859 -> v2 (claim development) 0.0955 -> v3 (freq x sev, bagging) 0.1039 -> v4 (+ validated cost trend) 0.1119.

## Approach
| Step | What | Why (evidence) |
|---|---|---|
| Target | Claims matched to policy rows by `policy_id` + loss date in period; net of recoveries, repudiated = 0; per-claim cap at the 99.9th percentile for training | Heavy tail: top 0.1% of rows hold 20-32% of cost |
| Leak removal | `customer_service_contacts` and `renewal_decision` excluded | Contacts track the same-row claim count (corr 0.75; D2 0.64 vs 0.06 without) and are flat at 0.40 in 2025 test vs 0.5-0.75 in training |
| Claim development | Chain-ladder reporting-lag CDF per peril; training exposure x reported fraction | 2024 claims valued at 31-Dec-2024 are immature (motor ~80% reported); developed 2024 motor cost/exposure 430 vs 343 raw. LB +0.0096 |
| Model | LightGBM Poisson frequency x Gamma severity, strong regularisation, 5-seed bagging | Beat 3-seed Tweedie pure-premium in 6/6 line x holdout cases (buildings 2023 only marginally; see `blend_eval_out.txt`). Seed bagging helped Tweedie in 4/6 cases (not yet measured for freq x sev) |
| Governance | Automated leak, drift, format and sanity guardrails (`guardrails.py`, `tests/`) | Run before any submission is written |
| Cost trend | Freq + severity log-linear trend from train years only (`trend.py`), damped x0.75, applied to all lines | Buildings/contents were 7-12% under-predicted on both holdouts. A trend estimated on pre-holdout years only improves D2 in 4/4 buildings/contents cases (A/E 0.98-1.04 at x0.75). Motor trend is ~0 and sign-unstable, so its 2025 factor is ~0.996 (`trend_eval_out.txt`) |

Rejected ideas (all validated, see `*_out.txt`): vehicle `make_model` (300 levels; overfits), weather/history features (tiny, noisy),
`written_premium`/`discount_applied` (risk under market change), an ad-hoc multiplicative level scalar (motor is calibrated, A/E 0.99-1.00; buildings/contents are 7-12% under-predicted on both holdouts but that tracks cost trend, not a cap/level artefact - see Open issues).

## Open issues
- Trend factor (buildings x1.16, contents x1.10 for 2025) extrapolates ~2.7 years from the pooled training mean, further than the holdouts (1.5-2 years). Damping x0.75 hedges this; it is a judgement made on holdout calibration, not on the leaderboard. Contents then sits about 12% above the 2024 developed level, the least certain part.

## Reproduce
```
pip install -r requirements.txt
# put the Kaggle data in data/public/*.parquet
python exp.py            # builds cached feature frames (data/cache)
python val_preds.py      # validation predictions (2023 & 2024 holdouts)
python blend_eval.py     # model / blend / load decisions
python final_v3.py --seeds 5 --w-fs 1.0 --tag final   # writes outputs/submission_final.csv (guardrails run inside)
python -m pytest -q      # guardrail tests
```

## Files
`prep.py` target construction | `features.py` features and exclusions | `dev.py` claim development | `models.py` model zoo |
`guardrails.py` + `tests/` governance | `final_v3.py` final pipeline | `eda_profile.py`, `leakage_check.py`, `dev_explore.py`, `exp*.py` analysis
