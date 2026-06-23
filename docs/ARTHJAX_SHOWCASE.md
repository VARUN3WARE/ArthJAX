# ArthJAX — Synthetic Economy Simulator

**A GPU-accelerated macro sandbox with emergent crises, contagion, and a neural world model.**

> Use this document as the source for your LinkedIn PDF. Embed the PNG files from the same folder when exporting.

---

## 1. The Problem

Real economies are too complex and too consequential to experiment on directly.

- Central banks and regulators **cannot rerun 2008** or “what if credit markets seize up?” in the real world.
- Traditional macro models (DSGE) are elegant but often **too abstract** — few agents, few behaviors.
- Large agent-based models exist, but many are **slow, hard to reproduce**, or locked in research institutions.
- LLM-based “AI economies” are flashy, but behavior comes from **language models**, not transparent economic rules.

**What’s missing:** a **fast, open, runnable sandbox** where:

1. Hundreds of agents interact through **realistic rules** (not scripted crashes).
2. **Booms, busts, and contagion emerge** from the system itself.
3. A **world model** can learn those dynamics for faster forecasting and counterfactuals.

**ArthJAX** is built to fill that gap.

---

## 2. The Idea — ArthJAX

**ArthJAX** is a fully vectorized **agent-based macroeconomic simulator** written in **JAX**. It runs on a GPU, simulates **600 timesteps in seconds**, and trains a small **neural world model** on the trajectories it generates.

| Dimension | ArthJAX |
|-----------|---------|
| **Households** | 250 agents, 4 behavioral types (value, momentum, panic, saver) |
| **Firms** | 60 companies across 10 sectors |
| **Financial system** | Banks, deposits, loans, bad-loan tracking, credit stress |
| **Markets** | Stocks (log prices), commodities, FX, volatility clustering |
| **Policy** | Taylor rule (interest rates), Phillips curve (inflation ↔ unemployment) |
| **Shocks** | Oil, credit crunch, demand, monetary policy (structured, random) |
| **Contagion** | Multi-step propagation across a sector dependency network |
| **Compute** | JIT-compiled `lax.scan` — no Python loops over agents or time |
| **AI layer** | MLP world model trained on normalized state transitions |

**Open-source GitHub repository — coming soon.**

---

## 3. How It Works (Notebook Pipeline)

The notebook `improved_synthetic_economy.ipynb` runs top to bottom on **Kaggle GPU** (~7 minutes). Each section builds one layer of the economy.

### Part A — Setup & initialization

| Step | What happens |
|------|----------------|
| **Environment** | JAX 0.7+ on **CUDA GPU**; outputs saved to working directory |
| **State design** | PyTree dictionary: wealth, firm balance sheets, bank books, macro variables, contagion buffer |
| **Initialization** | Random but bounded starting conditions; 4 household behavioral types |

**Typical output:**

```
JAX version: 0.7.2
Devices: [CudaDevice(id=0)]
Households: 250 (4 behavioral types)
Companies: 60 across 10 sectors
Macro: interest_rate=0.0400, inflation=0.0200
```

### Part B — Agent & market dynamics (vectorized functions)

Each timestep updates the economy in order:

1. **Households** — consumption and investment by behavioral type; macro-sensitive income.
2. **Companies** — revenue from demand, debt service, leverage limits, order-book price impact.
3. **Banks** — deposits, loan book, bad loans, credit stress index, leverage.
4. **Markets** — stock prices with per-firm mean reversion, volatility clustering, commodities, FX.
5. **Shocks** — rare structured events (oil, credit, demand, policy).
6. **Contagion** — delayed shock propagation through sector network (3-step memory).
7. **Macro feedback** — Phillips curve + Taylor rule + business-cycle forcing.
8. **Metrics** — GDP, inflation, unemployment, Gini, sentiment, credit stress, etc.

All compiled with **`jit`**; full step is **`step_jit`**.

### Part C — Main simulation

| Parameter | Value |
|-----------|--------|
| Timesteps | 600 |
| Runtime (Kaggle GPU) | ~6–7 seconds |
| Speed | ~80–90 steps/sec |
| History length | 601 points (includes t=0) |

**Example final macro state (after 600 steps):**

| Metric | Typical value | Meaning |
|--------|---------------|---------|
| GDP | ~$750–$850 | Total firm revenue (aggregate output) |
| Stock index | ~94–100 | Average stock price (anchored near $100) |
| Inflation | ~8–10% | Above 2% target → policy response |
| Unemployment | ~2.5–11% | Cycles between low and crisis levels |
| Interest rate | ~4–9% | Central bank Taylor-rule response |
| Sentiment | ~0.1–0.5 | Market confidence (steps down under stress) |
| Gini (peak) | ~0.05–0.15 | Wealth inequality early in run |
| Bad loan ratio | → 0% | Early stress, then banking stabilizes |
| Bank leverage | ~2x | Low leverage at end of run |

### Part D — World model training

| Item | Detail |
|------|--------|
| Training data | 600 state transitions from 5 rollouts × 120 steps |
| State vector size | **684 dimensions** (all households, firms, macro vars) |
| Normalization | Z-score (prevents neural network blow-up) |
| Architecture | 3-layer MLP, 128 hidden units, ReLU |
| Training | 50 epochs, batch size 64, gradient clipping |
| Final loss | ~**0.85** normalized MSE (decreasing from ~2.5) |

**Comparison rollout (200 steps):**

| Metric | Value |
|--------|--------|
| Macro-only prediction error | ~**21–32%** (interest, inflation, unemployment, sentiment) |
| Full-state error | Higher (~60–85%) — 684 variables is much harder |

The world model is an **early prototype** — it captures broad macro direction but not every detail.

---

## 4. Emergent Behavior (What the Simulation Produces)

Without scripting a “crash event,” the model still exhibits:

| Phenomenon | What we observe |
|------------|-----------------|
| **GDP shocks** | Multiple step-downs in output when conditions tighten |
| **Credit stress spike** | Sharp pulse (~4–5%) around mid-simulation, aligned with GDP drop |
| **Unemployment crisis** | Spike to ~11%, later partial recovery |
| **Phillips curve loops** | Inflation vs unemployment traces a path over time (color = time) |
| **Sentiment collapse** | Step-wise decline from neutral (0.5) toward pessimism |
| **Volatility clustering** | Macro volatility spikes during transitions, then calms |
| **Inequality dynamics** | Gini peaks early (~0.15), then compresses as wealth disperses |

These are **emergent** — they arise from agent rules, feedback loops, and shocks, not from a single “crash” function.

---

## 5. Visual Outputs — Chart Guide

Embed these images in your PDF. All files live in the project folder.

### Figure 1 — `macro_evolution_v2.png`  
**ArthJAX · Synthetic Economy Dynamics** (9-panel dashboard)

Dark-theme overview of the full economy over ~575 steps (after burn-in).

| Panel | What to look for |
|-------|------------------|
| **GDP** | High plateau then **stair-step declines** — output contractions |
| **Inflation** | Rises to ceiling (~10%), dips mid-run, volatile late phase |
| **Unemployment** | Low baseline, **spike to ~11%** mid-simulation, recovery toward end |
| **Interest rate** | Policy rate rises to ~9%, dip during crisis, returns up |
| **Stock index** | Oscillates ~95–102 — realistic band, not stuck at a ceiling |
| **Gini** | Early inequality, then decay (wealth spreads) |
| **Sentiment** | Step-down from 0.5 → 0.1 — confidence erodes |
| **Credit stress** | Near zero most of the time + **one sharp contagion spike** |
| **Bad loan ratio** | Early decay from ~3% to zero |

**Use for:** LinkedIn carousel slide 2 — “full economy dashboard.”

---

### Figure 2 — `linkedin_hero.png`  
**Emergent Macro Dynamics from 250 AI Agents** (hero chart)

Single chart for the **main LinkedIn post image**.

- **Blue — GDP ($):** Stable boom (~$1,100) then stepped bust toward ~$800.
- **Green — Stock index:** Volatile but bounded; market “noise” throughout.
- **Orange dashed — Unemployment (%):** Crisis arc — low → spike ~11% → recovery.

**Story in one image:** The economy booms, then **contagion and policy feedback** drive a mid-cycle crisis and partial recovery — all from agent interactions.

**Use for:** Primary LinkedIn post thumbnail.

---

### Figure 3 — `boom_bust_v2.png`  
**Emergent Boom-Bust Cycles & Macro Contagion**

Four analytical panels:

| Panel | Insight |
|-------|---------|
| **GDP vs Credit Stress** | Credit stress **spikes exactly when GDP steps down** — financial contagion |
| **Sentiment cycles** | Red shaded pessimism zone; sentiment falls in phases |
| **Phillips curve** | Scatter colored by time — economy **moves through regimes** (not a static curve) |
| **Macro volatility** | Huge volatility burst at crisis; rate vol stays lower |

**Use for:** “Research depth” slide — shows contagion and Phillips curve for technical audience.

---

### Figure 4 — `world_model_loss_v2.png`  
**World Model Training Convergence**

- Y-axis: normalized MSE (log scale).
- Loss drops from ~**2.5** → ~**0.8** over 50 epochs.
- Proves the neural network **learns** simulator dynamics (not random).

**Use for:** “AI layer” slide — connects simulation to machine learning.

---

### Figure 5 — `real_vs_learned_v2.png`  
**Neural World Model · Real vs Learned Economy**

Compares **simulator** (solid blue) vs **learned model** (dashed orange) over 200 steps:

| Variable | Simulator | World model |
|----------|-------------|-------------|
| Interest rate | Smooth rise to 9%, stable | Noisy; captures level roughly |
| Inflation | Smooth rise to 10% | Oscillates; mid-run instability |
| Unemployment | Drop to ~2.5% plateau | Noisy; underestimates crisis plateau |
| Sentiment | Clean step-down | Noisy; below real sentiment |

**Honest takeaway:** ~**25–32% macro error** — promising for a small MLP on 600 samples; **not** production forecasting yet.

**Use for:** “Future work” slide — world models as next step.

---

## 6. Why This Matters

### For policy & risk
- **Stress-test** credit, unemployment, and inflation interactions without real-world cost.
- Study **contagion timing** — shocks that hit credit first, then GDP.

### For machine learning
- **World models** need environments. ArthJAX is a **differentiable, fast** macro environment.
- Bridge between **agent-based economics** and **neural dynamics learning**.

### For builders
- **One notebook**, Kaggle GPU, reproducible charts.
- **JAX** = same stack as modern ML research (GPU, JIT, `lax.scan`).

### How ArthJAX differs from other projects

| Approach | Examples | ArthJAX |
|----------|----------|---------|
| LLM agents | Emergence World, AI Economy Simulator | Rule-based physics, no API costs |
| RL grid economies | AI Economist, EconoJax | Macro banking + Phillips + contagion |
| DSGE solvers | Equilibrium (JAX) | Emergent ABM, not equation-only |
| Bank stress tests | firesale_stresstest | Full macro loop + world model |

---

## 7. Key Numbers (At-a-Glance)

```
SCALE          250 agents · 60 firms · 10 sectors · 600 timesteps
SPEED          ~80 steps/sec on GPU · full run ~7 sec
GDP            ~$750–$1,100 range · step-down shock events
STOCKS         ~95–102 index · volatile but realistic
INEQUALITY     Gini peak ~0.15
CREDIT STRESS  Spike ~4–5% at contagion event
WORLD MODEL    ~0.85 training loss · ~25–32% macro prediction error
OUTPUTS        5 publication-style charts (dark theme)
```

---

## 8. Reproduce It Yourself

1. Upload `improved_synthetic_economy.ipynb` to **Kaggle**.
2. **Settings → Accelerator → GPU**.
3. **Run All**.
4. Download PNGs from the output folder.

Files generated:

- `macro_evolution_v2.png`
- `boom_bust_v2.png`
- `linkedin_hero.png`
- `world_model_loss_v2.png`
- `real_vs_learned_v2.png`

---

## 9. What’s Next (Open Source)

Planned for the public GitHub release:

- Clean README with install instructions
- Comparison to EconoJax / AI Economist
- Tunable parameters (shock frequency, agent mix)
- Larger world model (Flax/JAX)
- Counterfactual mode: “what if oil shock at t=200?”

**Follow for the repo drop.**

---

## 10. Suggested LinkedIn Narrative (Copy-Paste)

**Hook (problem):**  
We can’t stress-test a real economy. But we can build one.

**What I built:**  
ArthJAX — a GPU macro sandbox: 250 behavioral agents, 60 firms, banking contagion, Phillips curve + Taylor rule. Booms and busts **emerge** — no scripted crash.

**Proof:**  
600 timesteps in ~7 seconds. Credit stress spikes. GDP steps down. Unemployment hits ~11%. Then a neural **world model** learns those dynamics (~30% macro error, early prototype).

**CTA:**  
Open-sourcing on GitHub soon. Charts below.

**Hashtags:**  
`#MachineLearning #JAX #Economics #AI #OpenSource #Quant #AgentBasedModel`

---

## Appendix — Notebook Cell Map

| Cells | Content |
|-------|---------|
| Intro markdown | Project overview & Kaggle instructions |
| Setup | JAX import, GPU check, plot theme |
| Configuration | `init_state()` — agents, firms, banks, macro |
| Households | 4 behavioral types, vectorized |
| Companies | Revenue, debt, order-book prices |
| Banks | Credit stress, bad loans, leverage |
| Markets | Volatility clustering, mean reversion |
| Shocks | Oil, credit, demand, policy |
| Contagion | Sector network, delayed propagation |
| Macro feedback | Phillips + Taylor + cycles |
| Metrics | GDP, Gini, 21 indicators |
| Step + simulate | `step_jit` + `lax.scan` × 600 |
| Visualizations | Dashboard, boom-bust, hero chart |
| World model | Collect data, train MLP, loss plot |
| Validation | Real vs learned rollout |
| Summary | Stats + file list |

---

*Document generated for ArthJAX project showcase. Pair with PNG assets in this folder for PDF export.*
