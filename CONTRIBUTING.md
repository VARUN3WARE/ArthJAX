# Contributing to ArthJAX

Thanks for your interest — contributions are welcome whether you're fixing a bug, adding a scenario, improving docs, or extending the world model.

ArthJAX is at **v1.1.0** — core sim, macro world model v2, scenarios, benchmarks, CI, and a [live Kaggle demo](https://www.kaggle.com/code/varunraosfanlkan/arthjax-gpu-macro-abm-world-model).

**New here?** Start with issue [#Start here](https://github.com/VARUN3WARE/ArthJAX/issues?q=is%3Aissue+is%3Aopen+label%3Acommunity+in%3Atitle+%22Start+here%22) or browse [`good first issue`](https://github.com/VARUN3WARE/ArthJAX/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) tasks.

---

## How to contribute (fork + PR)

`main` is **maintainer-merge only**. Outside contributors:

1. **Fork** [VARUN3WARE/ArthJAX](https://github.com/VARUN3WARE/ArthJAX)
2. Create a branch on your fork
3. Install dev deps and run tests (below)
4. Open a **Pull Request** to `main` — do not push directly to the upstream repo

We aim to review PRs within **48–72 hours**.

### What we merge

- `pytest tests/ -q` passes (CI must be green)
- Small, focused diffs — one concern per PR
- Uses `EconomyConfig` patterns; no new global `NUM_*` constants
- Honest docs — no claims of real GDP forecasting

### What we won't merge

- Huge refactors without a prior issue
- Secrets, `.venv`, or large generated outputs
- Changes under `.local_dev/` (maintainer-local only)

---

## Getting started

```bash
git clone https://github.com/YOUR_USER/ArthJAX.git
cd ArthJAX
git remote add upstream https://github.com/VARUN3WARE/ArthJAX.git
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest tests/ -q
python scripts/run_simulation.py --steps 100
```

### CPU vs GPU

- **CI and most tests** run on **CPU** (`JAX_PLATFORM_NAME=cpu`).
- Install JAX with CUDA only if you need local GPU speed — see the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html).
- **No GPU?** Use the [Kaggle GPU demo](https://www.kaggle.com/code/varunraosfanlkan/arthjax-gpu-macro-abm-world-model) (Run All).

---

## What to work on

Pick from the [Project board](https://github.com/users/VARUN3WARE/projects) or filtered issues:

- [`good first issue`](https://github.com/VARUN3WARE/ArthJAX/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) — scoped for newcomers
- [`help wanted`](https://github.com/VARUN3WARE/ArthJAX/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22) — slightly harder, unowned
- [GitHub Issues](https://github.com/VARUN3WARE/ArthJAX/issues) — roadmap and community backlog

Issues labeled **`maintainer`** are Phase 2 core work — ask before starting.

Open an issue before **large** features so we don't duplicate work.

---

## Code guidelines

- **Config:** use `EconomyConfig` fields — avoid new global `NUM_*` constants.
- **Style:** match surrounding modules; small focused diffs.
- **JAX:** keep agent loops vectorized; use `make_step_jit(cfg)` for compiled steps.
- **World model:** train on normalized states; clip rollouts; test for finite loss.
- **Plots:** reuse `arthjax.viz.style` for consistent dark-theme charts.

---

## Commits & pull requests

**Commit subject:** imperative, ~50 characters, no trailing period.

```
Add --list flag to run_scenario.py.

Print scenario preset names and descriptions without running sim.
```

**Before opening a PR:**

1. `pytest tests/ -q` passes
2. No secrets, no `.venv`, no large notebook outputs committed
3. PR description explains *why*, not just *what*
4. Link the issue: `Fixes #123` when applicable

**Branching:** contributors use `feature/your-topic` on their fork. Maintainers may use `feature/*` on the upstream repo for Phase 2 work.

---

## Project layout

```
arthjax/       Simulation, agents, world model, viz
scripts/       CLI entry points
tests/         Pytest smoke tests (tiny configs for speed)
docs/          Showcase and assets
notebooks/     demo.ipynb, kaggle.ipynb
```

---

## Maintainer rhythm

| Cadence | Action |
|---------|--------|
| Weekly | Triage issues, update Project board |
| Biweekly | Open 1–2 new `good first issue` items |
| On merge | Close linked issue, thank contributor |

See [docs/CONTRIBUTING_ISSUES.md](docs/CONTRIBUTING_ISSUES.md) if you are seeding issues.

---

## Conduct

Be respectful in issues and reviews. We're building in public; constructive feedback beats hype.

---

## Questions

Open a [GitHub issue](https://github.com/VARUN3WARE/ArthJAX/issues) with the `question` label, or comment on the [Start here](https://github.com/VARUN3WARE/ArthJAX/issues?q=is%3Aissue+is%3Aopen+label%3Acommunity+in%3Atitle+%22Start+here%22) issue.
