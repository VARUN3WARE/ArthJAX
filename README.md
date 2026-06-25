# ArthJAX

**GPU-accelerated agent-based macro simulator with neural world models — built in JAX.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![CI](https://github.com/VARUN3WARE/ArthJAX/actions/workflows/ci.yml/badge.svg)](https://github.com/VARUN3WARE/ArthJAX/actions/workflows/ci.yml)
[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/varunraosfanlkan/arthjax-gpu-macro-abm-world-model)

![Emergent macro dynamics from 250 agents](docs/assets/linkedin_hero.png)

---

## The problem

Real economies cannot be stress-tested in production. Central banks, researchers, and ML teams need a **fast, reproducible sandbox** where booms, credit crunches, and contagion can emerge from agent rules—not from scripted crashes.

## What ArthJAX is

ArthJAX simulates a synthetic economy on a GPU:

| | |
|---|---|
| **Agents** | 250 households (4 behavioral types) · 60 firms · 10 sectors |
| **Financial system** | Banks, credit stress, bad loans, leverage cycles |
| **Markets** | Stocks, commodities, FX, volatility clustering |
| **Macro policy** | Phillips curve · Taylor rule · structured shocks |
| **Contagion** | Multi-step propagation across a sector network |
| **AI layer** | Neural world model trained on simulation rollouts (v0.3) |

**600 timesteps in seconds** on GPU via JIT-compiled `lax.scan`.

> **v0.5:** [Live Kaggle demo](https://www.kaggle.com/code/varunraosfanlkan/arthjax-gpu-macro-abm-world-model) — GPU, Run All, no local setup.

---

## Run on Kaggle (GPU)

**Published notebook:** [ArthJAX GPU Macro ABM + World Model](https://www.kaggle.com/code/varunraosfanlkan/arthjax-gpu-macro-abm-world-model)

[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/varunraosfanlkan/arthjax-gpu-macro-abm-world-model)

1. Open the link → **Run All** (GPU + Internet already configured on the published kernel).
2. Download charts from the **Output** tab (`macro_evolution_v2.png`, `linkedin_hero.png`, etc.).

Fork from source: [notebooks/kaggle.ipynb](notebooks/kaggle.ipynb) · [notebooks/README.md](notebooks/README.md)

---

## Roadmap

| Version | Deliverable |
|---------|-------------|
| **v0.1** | Repo foundation, `EconomyConfig`, docs |
| **v0.2** | Core simulation in `arthjax/` + `scripts/run_simulation.py` ✓ |
| **v0.3** | World model training + visualization CLI ✓ |
| **v0.4** | Tests + CI smoke run on CPU ✓ |
| **v0.5** | Kaggle notebook + README badges ✓ |
| **v1.0** | Benchmarks, scenarios, research-ready API (see issues) |

---

## Quick start

```bash
git clone https://github.com/VARUN3WARE/ArthJAX.git
cd ArthJAX
python -m venv .venv
source .venv/bin/activate
pip install -e .
python scripts/run_simulation.py --steps 600
python scripts/train_world_model.py --epochs 50 --plot
```

### GPU (optional)

Install JAX with CUDA support per [JAX installation docs](https://jax.readthedocs.io/en/latest/installation.html), then:

```bash
pip install -e .
```

---

## How ArthJAX compares

| Project | Focus | ArthJAX |
|---------|--------|---------|
| [AI Economist](https://github.com/salesforce/ai-economist) | 2D grid, RL tax policy | Macro banking + contagion + world model |
| [EconoJax](https://github.com/ponseko/econojax) | JAX + AI Economist + RL | Rule-based macro ABM, no RL required |
| LLM economies | Language-model agents | Differentiable physics, no API cost |
| DSGE solvers | Equation-based macro | Emergent multi-agent dynamics |

---

## Documentation

- [Full showcase & chart guide](docs/ARTHJAX_SHOWCASE.md) — problem, results, narrative for sharing
- [Contributing](CONTRIBUTING.md) — setup, PR guidelines, good first issues

---

## Contributing

Contributions are welcome — tests, scenarios, docs, and benchmarks are especially helpful right now. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and commit expectations.

If you're exploring the idea space: this is a **toy synthetic economy** for stress-testing and research prototypes, not a production forecasting tool. Issues and honest feedback are appreciated.

---

## Project structure

```
arthjax/          # Python package (simulation logic in v0.2+)
scripts/          # CLI entry points
notebooks/        # Demo & Kaggle notebooks
docs/             # Showcase and assets
tests/            # Test suite (v0.4)
```

---

## License

MIT — see [LICENSE](LICENSE).

---

## Author

[VARUN3WARE](https://github.com/VARUN3WARE)
