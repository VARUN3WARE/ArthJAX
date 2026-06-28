# GitHub Project board setup (one-time, ~5 min)

The API token used for automation cannot create Projects (needs `project` scope). Create the board manually:

1. Go to https://github.com/users/VARUN3WARE/projects → **New project**
2. Title: **ArthJAX Community**
3. Add columns (Table or Board view):
   - Backlog
   - Good first issue
   - Ready
   - In progress
   - In review
   - Done
4. Add open **`community`** issues from https://github.com/VARUN3WARE/ArthJAX/issues
5. Put **`good first issue`** labeled items in the **Good first issue** column
6. Pin the **[Start here] Contributing to ArthJAX** issue on the Issues page

## Branch protection

`main` requires PR + passing CI (`test` job). Only the repo owner merges.

## Labels

| Label | Purpose |
|-------|---------|
| `community` | For outside contributors |
| `good first issue` | Newcomer-safe |
| `help wanted` | Harder, unowned |
| `maintainer` | Phase 2 core — not for random PRs |

See [docs/CONTRIBUTING_ISSUES.md](../docs/CONTRIBUTING_ISSUES.md).
