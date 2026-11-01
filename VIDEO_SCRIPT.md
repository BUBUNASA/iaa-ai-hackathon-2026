# 5-minute video script - Team NazaliBupa, IAA AI-Assisted Hackathon 2026

About 700 spoken words (~140 words per minute). **[Screen]** = what to show; the rest is the voice-over.

---

## 0:00-0:30 The problem
**[Screen]** Title slide: "Pricing a known market and a new one - with AI, and with evidence"

An insurer sells motor, home buildings and home contents cover. For every 2025 policy we predict the expected claim cost - the risk premium,
the starting point of any price. Part 1 is the insurer's home market, with four years of data. Part 2 is a new market it entered two years ago
and is now growing fast through a broker. Throughout, we used one rule: a change goes in only if it improves the score on two separate holdout tests.

## 0:30-1:20 Trust before accuracy: two data traps
**[Screen]** Bar chart: validation D2 0.06 vs 0.64 with `customer_service_contacts`

Our first model looked brilliant: a score of 0.64 instead of 0.06. The reason was one field - customer service contacts. People call because they
have claimed, so that field is the answer, not a predictor. We removed it and added an automated test so it can never come back.

**[Screen]** Histogram of `exposure`: spike at 365

The second trap was quieter. About 0.4% of policies recorded their length in days - 365 instead of 1 year - adding a quarter of phantom cover.
We found it by asking why a gain on our holdouts did not show up on the leaderboard. Exposure is now rebuilt from the policy dates,
and a guardrail blocks any submission if it happens again. The same error was in the Part 2 data and was caught on the first load.

## 1:20-2:10 Part 1: four validated steps
**[Screen]** Waterfall: 0.086 -> 0.096 -> 0.104 -> 0.112 -> 0.113

Claims are valued at the end of 2024, so recent claims are still being reported. Modelling that reporting delay was our biggest single gain.
We then split each line into claim frequency and claim size, added a cost trend tested only on earlier years, and built separate models per peril
for home cover. Each step had to win on both the 2023 and 2024 holdouts. Our final private score was 0.126, rank 43.

## 2:10-3:20 Part 2: what changed, and how we adapted
**[Screen]** Comparison table: 18x fewer claims; 76% new business; motor claims more frequent, half the size; buildings fire = 45% of cost

The new market has eighteen times fewer claims, and three quarters of next year's customers are brand new. Car claims happen more often but cost about
half as much. Buildings costs swing with a handful of fires. A model trained only on this data cannot learn much - its claim-size model stops after one step.

**[Screen]** Four variants and the holdout table (emerging-only far behind on every line)

So we let the home market supply the structure and the new market correct it. We compared four ways of doing that on two holdouts - next year,
and unseen policies - and every one beat the emerging-only model, on every line. We blended the three transfer methods, choosing on the worst case
rather than the best.

**[Screen]** "Tested and rejected" table

Then we tried four more ideas. A new-business loading, a borrowed trend and per-peril models - our best Part 1 step - all failed:
with this little data, finer splits learn noise. Only the insurer's own pricing fields passed, for motor and buildings.

## 3:20-4:00 The business answer
**[Screen]** Office premium build-up for one reference policy, then the 2.1 / 2.3 / 4.1 ratios

Adding expenses, commission, tax, catastrophe reinsurance and profit to our claim cost gives the price the insurer needs. For the reference policies
that is about 2 to 4 times what is being charged. The book bought from the broker is under-priced. That is also why we submitted two models:
version 2 uses the current prices and forecasts slightly better, but a repricing built on it would be anchored to the price it replaces.
We recommend version 1 for pricing decisions.

## 4:00-4:40 How we used AI - and where we overruled it
**[Screen]** AI contribution vs human review table

AI wrote the analysis, the models, the guardrails and 31 automated tests, and it traced the exposure error. Every decision was backed by a result
file a person reviewed. Three of the AI's own ideas looked convincing and failed validation, and we dropped them. When AI-written tests were wrong,
we fixed them on evidence rather than loosening them to pass.

## 4:40-5:00 Limits and close
**[Screen]** Limitations list; repo link

The biggest uncertainties are the new market's buildings level, driven by a few fires, and a book that is three quarters new customers.
Everything is in the repository and reproducible from one command list. Thank you.
