# Personal settings: API keys and accounts

Everything that belongs to one person (API keys, the Weights & Biases team, ...) stays **outside the
repository**. The repo only holds the template [`.env.example`](../.env.example), which lists every
setting with no values. The `bss-settings` package (`src/bss_rebalancing/settings`, no dependencies)
reads the settings. Issue #176.

## Set up

```bash
pip install -e .                 # or: pip install -e src/bss_rebalancing/settings
bss-settings init                # creates ~/.config/bss-rebalancing/.env (mode 600) from .env.example
$EDITOR ~/.config/bss-rebalancing/.env
bss-settings check               # which settings are set and where from; values are never printed
```

Fill in only what you use. Empty values count as unset.

| Setting | Used by | Notes |
|---|---|---|
| `TOMTOM_API_KEY` | `bss-scenario traffic collect` | traffic data (issue #17) |
| `GOOGLE_MAPS_API_KEY` | travel-time cross-check | issue #58 |
| `WANDB_API_KEY` | `bss-train`, `train_grid` with `--wandb` | or `wandb login` once |
| `WANDB_ENTITY` | same | your W&B user or team; empty means your W&B default |
| `WANDB_PROJECT` | same | default `bss-rebalancing` |
| `WANDB_MODE` | same | `online` (default) or `offline` |

Command-line flags (`--wandb-entity`, `--wandb-project`, `--wandb-mode`) override the settings.

## Where settings are read from

Highest priority first:

1. **The environment**: `export NAME=value`, or `NAME=value command` for one run.
2. **`./.env`** in the directory you run from: per checkout, git-ignored.
3. **The user file**: `$BSS_ENV_FILE` if set, otherwise `$XDG_CONFIG_HOME/bss-rebalancing/.env`
   (default `~/.config/bss-rebalancing/.env`).

Files hold `NAME=value` lines. `export`, quotes and `#` comments are allowed. The same file can
also be `source`d by shell scripts, such as a cron job. Keep it private: `chmod 600`.
`bss-settings check` warns if others can read it.

## What is git-ignored

`.env`, `.env.*`, `*.env` (except `.env.example`), `secrets*.sh`, `.netrc` and `wandb/`. A test
(`tests/test_settings.py`) fails if a tracked file assigns a real-looking value to a key setting.
GitGuardian also scans every PR.

## Adding a setting

1. Add it to `KNOWN` in `src/bss_rebalancing/settings/src/bss_settings/__init__.py` (name,
   description, whether it is secret, what uses it).
2. Add `NAME=` with a comment to `.env.example`. A test checks that the two lists agree.
3. Read it in code:

   ```python
   import bss_settings
   key = bss_settings.require("MY_API_KEY")       # clear error saying where to put it if missing
   value = bss_settings.get("MY_OPTION", "default")
   bss_settings.apply_to_environ(["MY_API_KEY"])   # for libraries that read os.environ themselves
   ```

Never print, log or save a secret value, and never put one in a default, test, notebook or doc.
