# Contributing

Every change is tracked: **issue → branch → pull request → review → merge**.

## Branches

| Branch | Purpose |
|--------|---------|
| `main` | Released versions only (tagged). Updated from `dev` through a release PR. |
| `dev` | Integration branch. Everything lands here through a PR with green CI. |
| `<type>/<issue>-<slug>` | Work branches, one per issue, e.g. `feat/22-event-engine`, `fix/9-reset-seed`, `docs/4-sim-v2-spec`. |

Types: `feat`, `fix`, `docs`, `refactor`, `perf`, `test`, `ci`, `data`.

## Issues

- Every piece of work starts from an issue (templates: bug, feature, task).
- Labels: one `type:`, one `area:`, a `priority:`; `status: blocked` when waiting on something external.
- Milestones follow the simulator v2 plan (`docs/sim-v2/SPEC.md` §15). Bugs in the frozen thesis packages use `area: legacy`.

## Commits

Conventional style, imperative mood, referencing the issue:

```
feat(sim): add discrete-event engine with deterministic RNG streams (#22)

Body explaining what and why, wrapped at ~72 characters.
```

Scopes: `scenario`, `sim`, `gym`, `policies`, `legacy`, `ci`, `docs`.

## Pull requests

- Target `dev`; title in the same style as commits; body from the template, with `Closes #N`.
- CI must be green. Keep PRs focused on one issue.
- Merged with **squash**, so `dev` has one commit per PR; the branch is deleted after merge.

## Releases

`dev` → `main` through a release PR, then a tag `vX.Y.Z` (semantic versioning) with release notes
generated from the merged PRs.

## Local checks

```bash
pip install -e . pytest
pytest -q
```

## Data and secrets

- Raw and processed data live under `data*/` (git-ignored). Never commit data, models or results.
- API keys (e.g. `TOMTOM_API_KEY`) are read from environment variables, never committed.
