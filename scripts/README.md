# Scripts

All scripts save charts to **`plots/`** by default when using `--plot`.

| Script | Description |
|--------|-------------|
| `run_simulation.py` | Run macro ABM simulation (`--steps`, `--seed`, `--plot`) |
| `run_scenario.py` | Named stress scenarios (`--scenario`, `--list`, `--plot`) |
| `run_benchmarks.py` | Stylized facts + forecast comparison (`--steps`, `--plot`) |
| `run_counterfactual.py` | Same-seed policy diff (`--policy-a`, `--policy-b`, `--plot`) |
| `run_monte_carlo.py` | Multi-seed stress stats (`--scenario`, `--runs`) |
| `train_world_model.py` | Train neural world model (macro or full-state) |

```bash
python scripts/run_simulation.py --steps 600 --seed 42
python scripts/run_scenario.py --scenario credit_crunch --steps 600
python scripts/run_counterfactual.py --policy-a baseline --policy-b hawkish --plot
python scripts/run_monte_carlo.py --scenario credit_crunch --runs 20
```
