# ArthJAX public roadmap

Current release: **v1.0.0** — research-ready: scenarios, benchmarks, macro world model v2, CI.

Track work via [GitHub Issues](https://github.com/VARUN3WARE/ArthJAX/issues).

---

## Shipped

| Version | Highlights |
|---------|------------|
| v0.1 | Repo foundation, `EconomyConfig`, docs |
| v0.2 | Core simulation + `run_simulation.py` |
| v0.3 | World model training + visualization |
| v0.4 | Pytest + GitHub Actions CI |
| v0.5 | Kaggle notebook, Medium essay linked from README |
| v0.6 | Named crisis scenarios + `run_scenario.py` |
| v0.7 | Stylized-facts benchmarks + `docs/METHODS.md` |
| v0.8 | World model v2 (macro-only, multi-step loss) |
| v1.0 | Documented benchmarks, expanded CI, stable API |

---

## v1.0 criteria (met)

- Kaggle notebook runs end-to-end on a fresh fork
- 6+ stylized facts documented with pass/fail ranges ([BENCHMARKS.md](BENCHMARKS.md))
- No NaN on default 600-step simulation (CPU)
- World model macro rollout error below 25% on standard eval seed (~15%)
- `CONTRIBUTING.md` + [METHODS.md](METHODS.md) complete

---

## Longer term (post v1.0)

- Policy counterfactual search
- Calibration hooks (coarse fit to macro series)
- Larger agent counts via JAX sharding
- PyPI publish (`pip install arthjax`)

Contributions welcome — see [CONTRIBUTING.md](../CONTRIBUTING.md).
