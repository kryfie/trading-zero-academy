from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    return yaml.safe_load((ROOT / name).read_text(encoding="utf-8"))


def test_gen3_changes_architecture_but_preserves_world():
    g2 = load("config.yaml")
    g3 = load("config_generation3.yaml")
    assert g3["project"]["generation"] == 3
    assert g3["project"]["experiment_id"] == "GEN3_SEQUENCE_MEMORY_V1"
    assert g3["market"] == g2["market"]
    assert g3["world_rules"] == g2["world_rules"]
    assert g3["action_space"] == g2["action_space"]
    assert g3["evaluation"] == g2["evaluation"]
    for key in ["learning_rate", "n_steps", "batch_size", "gamma", "gae_lambda", "ent_coef", "clip_range", "policy_layers", "master_candidate_streak"]:
        assert g3["learner"][key] == g2["learner"][key]
    assert g3["learner"]["algorithm"] == "RecurrentPPO"
    assert g3["learner"]["policy"] == "MlpLstmPolicy"
    assert g3["learner"]["lstm_hidden_size"] == 256
    assert g3["learner"]["n_lstm_layers"] == 1
    assert g3["learner"]["target_total_timesteps"] == 1_000_000_000
