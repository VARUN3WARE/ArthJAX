# Writing good contributor issues (maintainer guide)

Use this when seeding **community** issues on the [Project board](https://github.com/users/VARUN3WARE/projects). Keep **maintainer** Phase 2 work labeled `maintainer` and off the Good first column.

**First time?** Create the Project board via [.github/PROJECT_SETUP.md](../.github/PROJECT_SETUP.md) (~5 min; requires GitHub UI).

---

## GOOD checklist

Every seed issue should have:

- **G**oal — one sentence outcome
- **O**ut of scope — what not to touch
- **O**ne proof — test name or CLI command that proves done
- **D**iff size — prefer &lt;200 lines

---

## Labels

| Label | When to use |
|-------|-------------|
| `community` | Opened for outside contributors |
| `good first issue` | Safe for newcomers, scoped, documented |
| `help wanted` | Harder but unowned |
| `maintainer` | Your Phase 2 core — not for random PRs |
| `documentation` / `tests` / `benchmarks` / `scenarios` | Area |

**Rule:** Close issues only when a PR merges (or explicit wontfix). Do not close because you implemented it solo.

---

## Project board columns

| Column | Contents |
|--------|----------|
| Backlog | Ideas, not ready yet |
| Good first issue | Ready for newcomers |
| Ready | Spec clear, can start today |
| In progress | Assignee / WIP |
| In review | Open PR linked |
| Done | Merged — then close issue |

---

## Maintainer rhythm

| Cadence | Action |
|---------|--------|
| Weekly (~15 min) | Triage labels, move cards, reply to questions |
| Biweekly | Open 1–2 new `good first issue` items |
| On merge | Close linked issue, thank contributor |
| On release | Update `docs/ROADMAP.md`, log in `.local_dev/PROGRESS.md` |

---

## Issue buckets (rotate)

- Tests — NaN guards, shape checks
- Docs — README examples, glossary
- Scenarios — CLI flags, notebook cells
- Benchmarks — CSV export, seed docs
- Viz — plot polish
- CI — smoke artifacts

Mark ~40% as `good first issue`, rest as `help wanted`.
