# ArthJAX methods & limitations

ArthJAX is a **toy synthetic economy** for stress-testing and research prototypes — not a production forecasting system.

---

## Agent model

| Layer | Description |
|-------|-------------|
| **Households** (250) | Four behavioral types (value, momentum, panic, saver) with type-specific consumption and risk |
| **Firms** (60) | Cash, debt, log stock price; sector assignment; leverage-driven distress |
| **Banks** | Deposits, loans, bad loans, leverage; credit stress index |
| **Markets** | Stocks, commodities, FX; mean-reverting prices with volatility clustering |
| **Macro policy** | Taylor rule (interest rate), Phillips curve (inflation), unemployment dynamics |
| **Shocks** | Random demand/oil/sentiment shocks with configurable probability |
| **Contagion** | Multi-step propagation across sector network |

All agent updates are **vectorized in JAX** and compiled via `make_step_jit(cfg)`.

---

## Key equations (simplified)

**Taylor rule (interest rate):**

\[
r_t = r_{t-1} + \alpha_r \cdot (\pi_t - \pi^*) + \alpha_u \cdot (u_t - u^*)
\]

**Phillips curve (inflation):**

\[
\pi_t = \pi_{t-1} + \beta \cdot (u^* - u_t) + \text{shock}_t
\]

**Credit stress:** rises with bad-loan ratio and bank leverage; feeds back into sentiment and lending.

See `arthjax/macro.py`, `arthjax/agents/`, and `arthjax/contagion.py` for full implementations.

---

## World model v2 (macro-only)

The neural world model predicts **4 macro variables** per step:

- `interest_rate`, `inflation`, `unemployment`, `sentiment_index`

Training uses macro-only transitions collected from simulation rollouts with diverse initial seeds. A **multi-step autoregressive loss** (horizon = 3) improves rollout stability.

**Full-state latent path (optional):** set `world_model_macro_only=False` and `world_model_use_latent=True` to train an encoder (684→32 on default config) plus latent MLP. Enabled via `EconomyConfig`; default remains macro-only.

**Evaluation seed:** `EconomyConfig.world_model_eval_seed = 200`

**Target:** macro rollout relative MAE **< 25%** on the standard eval seed (achieved ~14% on default config, CPU).

```bash
python scripts/train_world_model.py --epochs 80
python scripts/run_benchmarks.py
```

---

## Stylized facts

Six facts are computed from metrics history (see `arthjax/benchmarks/stylized_facts.py`):

| Fact | What it checks |
|------|----------------|
| GDP step-downs | Discrete output contractions |
| Vol clustering | Rolling GDP volatility varies |
| Phillips slope | Inflation–unemployment co-movement |
| Unemployment autocorr | Persistence (lag 5) |
| Credit stress spike | At least one elevated episode |
| Sentiment decline | No monotonic explosion |

Run: `python scripts/run_benchmarks.py --steps 600 --plot`

**Forecast comparison:** the CLI prints a table comparing full simulation (reference), persistence, AR(1), and world model macro MAE and runtime.

---

## Scenarios

Named presets adjust `EconomyConfig` and initial state (`arthjax/scenarios/presets.py`):

| Scenario | Effect |
|----------|--------|
| `credit_crunch` | Elevated bad loans, credit stress, tighter sentiment |
| `oil_shock` | High oil index, energy demand shock |
| `soft_landing` | Low shock rate, healthy banks |

```bash
python scripts/run_scenario.py --scenario credit_crunch --steps 600 --plot
```

---

## Policy counterfactuals (v1.2)

Taylor-rule knobs are exposed on `EconomyConfig`:

- `inflation_target`, `taylor_inflation_coef`, `taylor_unemployment_coef`, `taylor_rate_step`

Named presets in `arthjax/policy/presets.py`: `baseline`, `hawkish`, `dovish`.

Same seed + same initial state → compare macro paths under two policies:

```bash
python scripts/run_counterfactual.py --policy-a baseline --policy-b hawkish --plot
python scripts/run_monte_carlo.py --scenario credit_crunch --runs 20
```

---

## Limitations

- **Not calibrated** to real GDP, CPI, or unemployment series
- **Fixed agent counts** — scaling to 10k+ agents requires sharding (post v1.0)
- **Rule-based agents** — no RL or LLM behavior
- **World model** predicts macro tail only; micro state is frozen during learned rollout
- **CPU/GPU parity** — CI runs CPU; full GPU validation via Kaggle notebook

---

## References

- [Benchmark results](BENCHMARKS.md)
- [Contributing](../CONTRIBUTING.md)
