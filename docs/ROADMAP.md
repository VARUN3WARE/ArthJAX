# ArthJAX public roadmap

Current release: **v1.1.0** — Phase 1 backlog: benchmark plots, baseline table, latent encoder.

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
| v1.1 | Phillips/vol plots, forecast comparison table, latent encoder |

---

## Phase 1 (v1.1 — complete)

- [x] Volatility clustering test polish + benchmark tests
- [x] Phillips curve scatter plot (`run_benchmarks.py --plot`)
- [x] WM vs AR(1) vs full sim comparison table
- [x] Full-state latent encoder (684→32, optional via config)

---

## Longer term (Phase 2+)

- Policy counterfactual search
- Calibration hooks (coarse fit to macro series)
- Uncertainty bands (ensemble / dropout)
- Larger agent counts via JAX sharding
- PyPI publish (`pip install arthjax`)

Contributions welcome — see [CONTRIBUTING.md](../CONTRIBUTING.md).
