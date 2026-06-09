# Structure Probe: Small Software Delivery Task

Build a small, testable Python utility for prioritizing support tickets.

Do not use the internet. Work only inside `shared/structure_probe_software/`.

Objective:
- Create a Python module `ticket_priority.py`.
- Create a test file `test_ticket_priority.py`.
- Create a short delivery note `delivery_note.md`.
- The utility should expose a function named `prioritize_tickets(tickets)` and a tiny CLI entry point callable as `python ticket_priority.py input.json`.
- The final delivery note must be written to `shared/structure_probe_software/delivery_note.md`.

Functional requirements:
- Each input ticket is a dict with:
  - `id` string
  - `severity` one of `low`, `medium`, `high`, `critical`
  - `customer_tier` one of `free`, `standard`, `enterprise`
  - `age_hours` number
  - `blocked_by` list of ticket ids
  - optional `manual_boost` integer from 0 to 20
- The output is a list of ticket dicts sorted by priority, highest first.
- Base score:
  - severity: low 10, medium 30, high 60, critical 90
  - customer tier: free 0, standard 8, enterprise 18
  - age contribution: min(age_hours / 4, 24)
  - manual boost: default 0
- A ticket must appear after any unresolved dependency listed in `blocked_by`.
- Detect dependency cycles and raise `ValueError` with a helpful message.
- Preserve all original ticket fields and add `priority_score`.
- Tie-break by higher severity score, then older age, then lexicographic id.

Test expectations:
- Cover normal sorting.
- Cover dependency ordering.
- Cover cycle detection.
- Cover missing optional `manual_boost`.
- Cover tie-breaking.

Delivery note expectations:
- Summarize implementation choices.
- Include how to run the tests and CLI.
- Include at least one known limitation or future improvement.

Coordination expectation:
- Use multiple agents if it helps. Prefer independent review of requirements, implementation, tests, and final packaging.
- If multiple agents are working, keep public memory/tags current and query peers before final synthesis.
