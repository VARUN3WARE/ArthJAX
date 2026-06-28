# ArthJAX benchmark results

Last run: default `EconomyConfig`, 600 steps, seed 42, CPU.

Command: `python scripts/run_benchmarks.py --steps 600`

---

## Stylized facts (7/7 pass)

| Fact | Value | Range | Result |
|------|-------|-------|--------|
| GDP step-downs | 2.0 | 1.0–15.0 | PASS |
| Vol clustering | 6.39 | 5.0–500.0 | PASS |
| Phillips slope | -0.94 | -5.0–2.0 | PASS |
| Unemployment autocorr (lag 5) | 0.99 | 0.3–0.99 | PASS |
| Credit stress spike | 0.10 | 0.02–0.5 | PASS |
| Sentiment decline | 0.40 | -0.5–0.6 | PASS |
| Volatility autocorr (lag 3) | 0.76 | 0.2–0.99 | PASS |

The volatility autocorr fact uses rolling stock-index return volatility (since `avg_volatility` flatlines at `vol_min` in long runs).

---

## World model v2 (macro-only)

| Metric | Value |
|--------|-------|
| Eval seed | 200 (`EconomyConfig.world_model_eval_seed`) |
| Macro MAE | 0.0226 |
| Macro relative error | **15.1%** (target < 25%) |
| WM train time | ~2.6s (CPU) |
| WM rollout time | ~0.1s for 100 steps |
| Full sim time | ~1.5s for 600 steps |
| Speedup (rollout vs sim) | ~15× |

---

## AR(1) baseline

| Metric | Value |
|--------|-------|
| MAE (interest rate column) | 0.0188 |
| Horizon | 50 steps |

---

## Scenario smoke

```bash
python scripts/run_scenario.py --scenario credit_crunch --steps 600
python scripts/run_scenario.py --scenario oil_shock --steps 600
python scripts/run_scenario.py --scenario soft_landing --steps 600
```

Credit crunch shows higher initial credit stress and bad-loan ratio vs baseline (see `tests/test_scenarios.py`).

---

## Forecast comparison (v1.1)

| Method | Macro MAE | Rel % | Notes |
|--------|-----------|-------|-------|
| Full simulation | 0.0 | 0% | Reference (ground truth) |
| Persistence | ~0.012 | ~7% | Last-value baseline |
| AR(1) | ~0.012 | ~7% | Rolled forward per macro col |
| World model | ~0.023 | ~15% | Macro-only rollout |

Run: `python scripts/run_benchmarks.py --steps 600 --plot`

Plots: `phillips_scatter.png`, `vol_clustering.png`

---

```bash
pip install -e ".[dev]"
pytest tests/ -q
python scripts/run_benchmarks.py --steps 600
python scripts/run_benchmarks.py --quick --steps 80   # CI smoke
```

See [METHODS.md](METHODS.md) for methodology and limitations.
