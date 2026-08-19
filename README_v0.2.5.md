# Trading Zero Academy v0.2.5 — cache migration fix

Apply this patch on top of the current v0.2.4 Marathon repository.

What it fixes:
- restores the v0.2.2 Student #1 cache using the original cache path set;
- uses a separate stable Marathon cache namespace afterwards;
- refuses to start training if models/latest.zip was not restored;
- prints the restored Student timestep count before training.

Do not let the currently running broken Marathon reach its training step.
Cancel it first, apply this patch, commit/push, then launch Marathon again.
