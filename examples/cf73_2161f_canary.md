# CF-73 2161F Canary

Run a small no-cheat canary on CF-73 problem 2161F (SubMST).

Public files:
- `shared/cf73_top10/problems/2161F/metadata.json`
- `shared/cf73_top10/problems/2161F/statement.md`

Rules:
- Use only public files in `shared/cf73_top10`.
- Do not use web search, local judge, hidden tests, Polygon packages, official solutions, checkers, interactors, or generated-test commands.
- Shell is disabled.
- The bootstrap has already launched one coordinator and three generator peers for `problem:2161F`.

Canary goal:
- Verify that generator agents can produce artifact files, publish them in memory, and let the coordinator discover them through query.
- Do not run private evaluation in this canary.

Coordinator expectation:
- Query `group_id=cf73-2161F-gen0-canary`.
- Inspect `shared/cf73_top10/2161F/gen0/`.
- Write `shared/cf73_top10/2161F/canary_report.md` summarizing which candidate files exist, whether artifact_commit/file_write was used, and what remains incomplete.
