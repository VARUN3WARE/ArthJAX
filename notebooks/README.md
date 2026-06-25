# Notebooks

| Notebook | Where to run | Description |
|----------|--------------|-------------|
| [**Kaggle (live)**](https://www.kaggle.com/code/varunraosfanlkan/arthjax-gpu-macro-abm-world-model) | Kaggle GPU | Published demo — Run All, download charts |
| [`kaggle.ipynb`](kaggle.ipynb) | Kaggle GPU | Source notebook (install from GitHub) |
| [`demo.ipynb`](demo.ipynb) | Local | Thin wrapper after `pip install -e .` |

## Run on Kaggle

**Live kernel:** https://www.kaggle.com/code/varunraosfanlkan/arthjax-gpu-macro-abm-world-model

Open → **Run All** → download PNGs from **Output**.

To republish or fork from repo:

1. Upload [`kaggle.ipynb`](kaggle.ipynb) or use [`kernel-metadata.json`](kernel-metadata.json) with the Kaggle API.
2. **Accelerator: GPU** · **Internet: On**
3. Outputs land in `/kaggle/working/`

## Local demo

```bash
pip install -e .
jupyter notebook notebooks/demo.ipynb
```

Legacy mega-notebook (`improved_synthetic_economy.ipynb`) stays local until retired.
