# Contributing to ArthJAX

Thanks for your interest — contributions are welcome whether you're fixing a bug, adding a scenario, improving docs, or extending the world model.

ArthJAX is early-stage (v0.4). The core sim runs; we're working toward public benchmarks and a Kaggle demo. Honest issues and small PRs are more valuable than huge rewrites.

---

## Getting started

```bash
git clone https://github.com/VARUN3WARE/ArthJAX.git
cd ArthJAX
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest tests/ -q
python scripts/run_simulation.py --steps 100
```

Tests run on **CPU** by default. For GPU, install JAX with CUDA per the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html).

---

## What to work on

Good first contributions:

- **Tests** — edge cases, shape checks, regression guards against NaN
- **Docs** — clarify agent rules, fix typos, add examples
- **Scenarios** — named stress presets (oil shock, credit crunch) with tests
- **Benchmarks** — stylized facts (volatility clustering, Phillips curve)
- **Notebooks** — Kaggle-friendly demo (see roadmap v0.5)

Check the [README roadmap](README.md#roadmap) for planned releases. Open an issue before large features so we don't duplicate work.

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
Add credit-crunch scenario preset.

Expose elevated bad-loan initialization and shock schedule via
EconomyConfig override and scripts/run_scenario.py.
```

**Before opening a PR:**

1. `pytest tests/ -q` passes
2. No secrets, no `.venv`, no large notebook outputs committed
3. PR description explains *why*, not just *what*

**Branching:** small fixes can target `main` directly. Larger features (new subsystems, refactors) should use `feature/your-topic`.

---

## Project layout

```
arthjax/       Simulation, agents, world model, viz
scripts/       CLI entry points
tests/         Pytest smoke tests (tiny configs for speed)
docs/          Showcase and assets
notebooks/     demo.ipynb (more coming)
```

---

## Conduct

Be respectful in issues and reviews. We're building in public; constructive feedback beats hype.

---

## Questions

Open a [GitHub issue](https://github.com/VARUN3WARE/ArthJAX/issues) with the `question` label, or reach out via the repo owner profile.
