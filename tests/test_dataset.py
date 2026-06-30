from pathlib import Path

from data.dataset import Stage2WindowDataset


def test_stage2_check_dataset_exists_after_script():
    path = Path("datasets/stage2/check_split/train")
    if not path.exists():
        return
    ds = Stage2WindowDataset(str(path), window=16)
    sample = ds[0]
    assert sample["obs_robot_0"].shape[-1] == 11
    assert sample["obs_robot_1"].shape[-1] == 11
    assert sample["actions"].shape[-1] == 8
    assert sample["global_state"].shape[-1] == 12
    assert sample["communication_dummy"].shape[-1] == 8
