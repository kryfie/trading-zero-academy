import numpy as np

from academy.recurrent_utils import recurrent_predict


class DummyModel:
    def __init__(self):
        self.calls = []
    def predict(self, obs, state=None, episode_start=None, deterministic=False):
        self.calls.append((obs, state, episode_start.copy(), deterministic))
        return 7, ("next-h", "next-c")


def test_recurrent_predict_carries_state_and_episode_start():
    model = DummyModel()
    action, state = recurrent_predict(model, np.array([1.0]), lstm_state=("h", "c"), episode_start=True, deterministic=True)
    assert action == 7
    assert state == ("next-h", "next-c")
    _, passed_state, starts, deterministic = model.calls[0]
    assert passed_state == ("h", "c")
    assert starts.shape == (1,)
    assert bool(starts[0]) is True
    assert deterministic is True
