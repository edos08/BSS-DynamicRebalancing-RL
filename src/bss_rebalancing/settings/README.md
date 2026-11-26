# bss-settings

Personal settings (API keys, tracking accounts, ...) for this project, kept **outside the
repository**. No dependencies.

```bash
pip install -e src/bss_rebalancing/settings
bss-settings init      # creates ~/.config/bss-rebalancing/.env (mode 600) from .env.example
$EDITOR ~/.config/bss-rebalancing/.env
bss-settings check     # which settings are set and where from (values are never printed)
```

Priority: the environment, then `./.env` (git-ignored), then the user file (`$BSS_ENV_FILE`, else
`$XDG_CONFIG_HOME/bss-rebalancing/.env`). In code:

```python
import bss_settings
key = bss_settings.require("TOMTOM_API_KEY")     # error explaining where to put it if missing
entity = bss_settings.get("WANDB_ENTITY")        # None if not set
bss_settings.apply_to_environ(["WANDB_API_KEY"]) # for libraries reading os.environ themselves
```

To add a setting: add it to `KNOWN` in `bss_settings/__init__.py` and to `.env.example` (a test
checks they agree). See [docs/configuration.md](../../../docs/configuration.md).
