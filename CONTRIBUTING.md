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
- Decisions, findings and progress are written as comments on the issue, so the reasoning stays with the work.

### Hierarchy

Issues are nested with GitHub sub-issues:

```
Roadmap: Simulator v2                      (type: epic)
└── Epic: <milestone>                      (type: epic, one per milestone, plus "Legacy maintenance")
    └── <issue>                            (feature / data / bug / ...)
        └── <task>                         (sub-issue, closed by one PR)
```

- Every new issue is attached to the epic of its milestone (or to the legacy epic).
- When work on an issue starts, it is broken into task sub-issues small enough for one PR each.
  Small issues that fit in one PR need no tasks.
- An epic or issue is closed when all its sub-issues are closed.

### Project board

All issues and PRs are on the project board *BSS Simulator v2* with a `Status` field:

| Status | When |
|--------|------|
| Todo | Issue created, work not started |
| In Progress | Branch created |
| In Review | PR open |
| Done | PR merged / issue closed |

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
- Before merging, a review comment on the PR states what was checked.
- GitHub closing keywords (`Closes #N`) only act on merges into the default branch (`main`). After
  a merge into `dev`, close the issue by hand with a comment naming the PR and the squash commit,
  e.g. `Done in #34 (squash d1e091f on dev)`.

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
