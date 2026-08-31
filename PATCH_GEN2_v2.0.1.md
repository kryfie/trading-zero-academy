# Generation 2 v2.0.1 — Python import-path fix

Fixes the first-run GitHub Actions failure:

`ModuleNotFoundError: No module named 'academy'`

No trading, reward, data, PPO, action-space, fee, slippage, funding, validation, or 500M protocol parameters are changed.

The patch only exposes the repository root to Python through `PYTHONPATH` in both Generation 2 workflows, so scripts launched as `python scripts/*.py` can import the local `academy` package.
