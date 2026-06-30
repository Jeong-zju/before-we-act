from pathlib import Path

from data.diagnostics import list_episodes, read_episode, validate_episode


def test_diagnostics_on_train_if_available():
    data_dir = Path("datasets/stage2/train")
    if not data_dir.exists():
        return

    episodes = list_episodes(data_dir)
    if not episodes:
        return

    ok, errors = validate_episode(episodes[0])
    assert ok, errors

    ep = read_episode(episodes[0])
    assert ep["actions"].shape[-1] == 8
    assert ep["obs_robot_0"].shape[-1] == 11
    assert ep["obs_robot_1"].shape[-1] == 11
    assert ep["object_pose"].shape[-1] == 3
    assert ep["global_state"].shape[-1] == 12
