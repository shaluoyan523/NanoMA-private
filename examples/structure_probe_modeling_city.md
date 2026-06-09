# Structure Probe: City Transit Modeling Report

Produce a compact mathematical modeling report for the fictional city **Lumen Bay**.

Do not use the internet. Use only the evidence below and any calculations you can derive from it.

Objective:
- Recommend which two transit corridor projects should be funded first.
- Define a transparent scoring or optimization model.
- Include assumptions, equations or pseudocode, sensitivity checks, and implementation notes.
- Write the final report to `shared/structure_probe_modeling/final_report.md`.
- Write any intermediate notes, calculations, and scratch files under `shared/structure_probe_modeling/`.

Problem context:
- Lumen Bay has a 92M capital budget for the first build phase.
- The city wants to maximize combined mobility, equity, and emissions benefits while keeping operating risk manageable.
- At least one selected corridor must serve a high-equity-need district.
- No selected pair may exceed the capital budget.
- A project with delivery risk above 0.70 requires an explicit mitigation plan.

Candidate corridors:

| ID | Corridor | Capex M | Annual riders M | Equity need 0-1 | CO2 reduction kt/year | Delivery risk 0-1 | Opex M/year |
|----|----------|---------|------------------|-----------------|-----------------------|-------------------|-------------|
| A | Harbor Loop BRT | 38 | 7.8 | 0.64 | 18 | 0.42 | 5.6 |
| B | West Ridge Light Rail Extension | 72 | 10.5 | 0.48 | 26 | 0.76 | 7.4 |
| C | South Market Rapid Bus | 29 | 6.1 | 0.82 | 14 | 0.35 | 4.1 |
| D | North Campus Connector | 44 | 5.2 | 0.36 | 11 | 0.31 | 3.8 |
| E | East Industrial Tram | 55 | 8.0 | 0.71 | 21 | 0.68 | 6.2 |
| F | Airport Express Upgrade | 46 | 4.8 | 0.22 | 9 | 0.28 | 4.9 |

Policy weights from council staff:
- Mobility: 40%
- Equity: 30%
- Emissions: 20%
- Cost discipline: 10%

Additional constraints and evidence:
- High-equity-need district threshold: equity need >= 0.70.
- Operating risk increases when combined opex exceeds 12.0M/year.
- South Market has the highest bus crowding complaints.
- East Industrial has a strong freight-adjacent jobs access argument, but schedule reliability is uncertain.
- West Ridge has the largest ridership and emissions upside, but land acquisition risk is high.
- Harbor Loop can share depots with existing fleet.
- Airport Express has political support but weak equity score.

Required final report sections:
- Recommendation.
- Model definition.
- Pairwise evaluation table.
- Sensitivity checks.
- Risk and mitigation notes.
- What evidence would change the recommendation.

Coordination expectation:
- Use multiple agents if it helps. Prefer independent checks for assumptions, calculations, and synthesis.
- If multiple agents are working, keep public memory/tags current and query peers before final synthesis.
