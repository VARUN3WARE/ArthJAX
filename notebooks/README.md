# Notebooks

| Notebook | Where to run | Description |
|----------|--------------|-------------|
| [`kaggle.ipynb`](kaggle.ipynb) | **Kaggle GPU** (recommended) | Full demo: sim + charts + world model |
| [`demo.ipynb`](demo.ipynb) | Local / Colab | Thin wrapper after `pip install -e .` |

## Run on Kaggle

1. Fork or upload [`kaggle.ipynb`](kaggle.ipynb) to Kaggle (use [`kernel-metadata.json`](kernel-metadata.json) if using the Kaggle API).
2. Settings → **Accelerator: GPU T4 x2** (or any GPU).
3. Settings → **Internet: On** (required to `pip install` from GitHub).
4. **Run All** — outputs land in `/kaggle/working/`.

After publishing, update the kernel URL in the root [README](../README.md).

## Local demo

```bash
pip install -e .
jupyter notebook notebooks/demo.ipynb
```

Legacy mega-notebook (`improved_synthetic_economy.ipynb`) stays local until retired.
