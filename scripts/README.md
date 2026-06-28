# Scripts

| Script | Description |
|--------|-------------|
| `run_simulation.py` | Run macro ABM simulation (`--steps`, `--seed`, `--plot`) |
| `run_scenario.py` | Named stress scenarios (`--scenario`, `--steps`, `--plot`) |
| `run_benchmarks.py` | Stylized facts + forecast comparison (`--steps`, `--plot`) |
| `train_world_model.py` | Train neural world model (macro or full-state) |

```bash
python scripts/run_simulation.py --steps 600 --seed 42
python scripts/run_scenario.py --scenario credit_crunch --steps 600
python scripts/run_benchmarks.py --steps 600 --plot
```
