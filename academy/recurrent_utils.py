from __future__ import annotations

import numpy as np


def recurrent_predict(model, obs, lstm_state=None, episode_start: bool = False, deterministic: bool = True):
    """Predict one action while explicitly carrying recurrent state.

    `episode_start=True` resets the recurrent state for the first observation of
    an episode. Omitting this state plumbing would reduce an LSTM policy to a
    sequence of effectively memory-reset decisions during evaluation.
    """
    starts = np.asarray([bool(episode_start)], dtype=bool)
    return model.predict(
        obs,
        state=lstm_state,
        episode_start=starts,
        deterministic=deterministic,
    )
