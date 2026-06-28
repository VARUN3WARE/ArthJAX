#!/usr/bin/env python3
"""Bootstrap community issues and comments for OSS dual-track plan."""
from __future__ import annotations

import json
import os
import re
import urllib.request

REPO = "VARUN3WARE/ArthJAX"
TOKEN = ""
for line in open(os.path.expanduser("~/.git-credentials")):
    m = re.search(r"https://[^:]+:([^@]+)@github.com", line)
    if m:
        TOKEN = m.group(1)
        break

BASE = f"https://api.github.com/repos/{REPO}"


def api(method: str, path: str, data: dict | None = None) -> dict:
    url = path if path.startswith("http") else f"{BASE}{path}"
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={
            "Authorization": f"token {TOKEN}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())


CLOSED_COMMENTS = {
    1: "Shipped in [v0.6.0](https://github.com/VARUN3WARE/ArthJAX/releases/tag/v0.6.0) — `credit_crunch` scenario preset and `run_scenario.py`. Closed as completed; see open **community** issues for new contributor tasks.",
    2: "Shipped in [v0.6.0](https://github.com/VARUN3WARE/ArthJAX/releases/tag/v0.6.0) — `oil_shock` scenario preset. See open **community** issues for new work.",
    3: "Shipped in [v1.1.0](https://github.com/VARUN3WARE/ArthJAX/releases/tag/v1.1.0) — volatility clustering tests in `tests/test_benchmarks.py`.",
    4: "Shipped in [v1.1.0](https://github.com/VARUN3WARE/ArthJAX/releases/tag/v1.1.0) — Phillips scatter via `run_benchmarks.py --plot`.",
    5: "Shipped in [v1.1.0](https://github.com/VARUN3WARE/ArthJAX/releases/tag/v1.1.0) — latent encoder path (`world_model_use_latent=True`).",
    6: "Shipped in [v1.1.0](https://github.com/VARUN3WARE/ArthJAX/releases/tag/v1.1.0) — forecast comparison table in `run_benchmarks.py`.",
    7: "Shipped in [v0.7.0](https://github.com/VARUN3WARE/ArthJAX/releases/tag/v0.7.0) / [v1.0.0](https://github.com/VARUN3WARE/ArthJAX/releases/tag/v1.0.0) — `docs/METHODS.md`.",
    8: "Shipped in [v1.0.0](https://github.com/VARUN3WARE/ArthJAX/releases/tag/v1.0.0) — CONTRIBUTING links to `docs/ROADMAP.md`.",
    9: "Merged via PR #9 — roadmap link fix. Thanks @JaleedAhmad!",
}

NEW_ISSUES = [
    {
        "title": "[Start here] Contributing to ArthJAX",
        "labels": ["community", "documentation"],
        "body": """## Welcome

ArthJAX is a **toy synthetic macro simulator** for stress-testing — not production GDP forecasting.

## How to contribute

1. Read [CONTRIBUTING.md](https://github.com/VARUN3WARE/ArthJAX/blob/main/CONTRIBUTING.md)
2. Pick an open issue labeled [`good first issue`](https://github.com/VARUN3WARE/ArthJAX/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
3. **Fork** the repo → branch → `pytest tests/ -q` → **Pull Request** (no direct push to `main`)

## Project board

Track work on the [ArthJAX Community Project](https://github.com/users/VARUN3WARE/projects) (columns: Backlog → Good first issue → Ready → In progress → In review → Done).

Maintainer aims to review PRs within **48–72 hours**.

## Questions

Comment here or open an issue with the `question` label.
""",
    },
    {
        "title": "Add --list flag to run_scenario.py",
        "labels": ["community", "good first issue", "scenarios"],
        "body": """## Goal
Print all scenario preset names and descriptions without running a simulation.

## Out of scope
Do not change scenario logic or presets.

## Acceptance criteria
- [ ] `python scripts/run_scenario.py --list` prints `baseline`, `credit_crunch`, `oil_shock`, `soft_landing` with descriptions
- [ ] `pytest tests/ -q` passes

## How to verify
```bash
python scripts/run_scenario.py --list
pytest tests/ -q
```

**Estimated difficulty:** S
""",
    },
    {
        "title": "Export benchmark comparison table to CSV",
        "labels": ["community", "good first issue", "benchmarks"],
        "body": """## Goal
Add optional CSV export of the forecast comparison table from `run_benchmarks.py`.

## Out of scope
No change to stylized-facts computation.

## Acceptance criteria
- [ ] `--csv path.csv` writes method, macro_mae, relative_error_pct, elapsed_sec, steps
- [ ] Smoke test or unit test for CSV row count

## How to verify
```bash
python scripts/run_benchmarks.py --quick --steps 80 --csv /tmp/bench.csv
pytest tests/ -q
```

**Estimated difficulty:** S
""",
    },
    {
        "title": "Document world model eval seed 200 in README",
        "labels": ["community", "good first issue", "documentation"],
        "body": """## Goal
One-line note in README that macro WM eval uses `EconomyConfig.world_model_eval_seed = 200`.

## Out of scope
No code changes to training.

## Acceptance criteria
- [ ] README mentions eval seed 200 and links to `docs/METHODS.md`

## How to verify
Read README diff; link resolves.

**Estimated difficulty:** S
""",
    },
    {
        "title": "Regression test: 600-step sim produces no NaN",
        "labels": ["community", "good first issue", "benchmarks"],
        "body": """## Goal
Add pytest that runs a short or tiny 600-step simulation and asserts all metrics are finite.

## Out of scope
Do not change simulation dynamics.

## Acceptance criteria
- [ ] New test in `tests/` using tiny config OR marked slow with reduced steps documented
- [ ] `pytest tests/ -q` passes in CI under 60s

## How to verify
```bash
pytest tests/ -q
```

**Estimated difficulty:** S
""",
    },
    {
        "title": "Add credit_crunch scenario cell to notebooks/demo.ipynb",
        "labels": ["community", "good first issue", "documentation"],
        "body": """## Goal
Add one notebook cell demonstrating `run_scenario` or in-notebook `credit_crunch` preset.

## Out of scope
Do not commit large notebook outputs.

## Acceptance criteria
- [ ] Cell runs with tiny config or documents CPU runtime
- [ ] Notebook JSON stays small

## How to verify
Run notebook top-to-bottom locally.

**Estimated difficulty:** M
""",
    },
    {
        "title": "Improve Phillips plot slope annotation",
        "labels": ["community", "help wanted", "benchmarks"],
        "body": """## Goal
Enhance `plot_phillips_scatter` legend/annotation for OLS slope readability.

## Out of scope
No change to stylized-facts pass/fail thresholds.

## Acceptance criteria
- [ ] Slope visible on chart; `run_benchmarks.py --plot` still works
- [ ] Plot smoke test passes

**Estimated difficulty:** S
""",
    },
    {
        "title": "Add CONTRIBUTING section on JAX CPU vs GPU",
        "labels": ["community", "help wanted", "documentation"],
        "body": """## Goal
Short CONTRIBUTING subsection: when CPU is enough, when to install CUDA jaxlib, Kaggle as GPU path.

## Acceptance criteria
- [ ] Links to JAX install docs and Kaggle kernel

**Estimated difficulty:** S
""",
    },
    {
        "title": "Typo sweep in docs/METHODS.md",
        "labels": ["community", "help wanted", "documentation"],
        "body": """## Goal
Fix typos, broken links, and clarify one agent rule paragraph in METHODS.md.

## Acceptance criteria
- [ ] No behavioral code changes
- [ ] PR lists what was clarified

**Estimated difficulty:** S
""",
    },
    {
        "title": "Add good first issues link badge in README",
        "labels": ["community", "help wanted", "documentation"],
        "body": """## Goal
Add shields.io or text link near top of README to open good-first issues filter.

## Acceptance criteria
- [ ] Link: `https://github.com/VARUN3WARE/ArthJAX/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22`

**Estimated difficulty:** S
""",
    },
    {
        "title": "CI workflow_dispatch benchmark artifact upload",
        "labels": ["community", "help wanted", "benchmarks"],
        "body": """## Goal
On manual CI dispatch, upload benchmark smoke output or stylized-facts summary as artifact.

## Out of scope
No GPU CI required.

## Acceptance criteria
- [ ] `workflow_dispatch` job uploads artifact
- [ ] Document in CONTRIBUTING or docs/BENCHMARKS.md

**Estimated difficulty:** M
""",
    },
]


def main() -> None:
    for num, body in CLOSED_COMMENTS.items():
        try:
            api("POST", f"/issues/{num}/comments", {"body": body})
            print(f"comment #{num}")
        except Exception as e:
            print(f"comment #{num} err", e)

    created = []
    for issue in NEW_ISSUES:
        try:
            res = api("POST", "/issues", issue)
            created.append(res["number"])
            print(f"created #{res['number']}: {issue['title'][:50]}")
        except Exception as e:
            print("create err", issue["title"][:40], e)

    print("created_ids", created)


if __name__ == "__main__":
    main()
