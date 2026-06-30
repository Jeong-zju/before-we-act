from pathlib import Path
import h5py
import torch
from torch.utils.data import Dataset


class DummyCarryDataset(Dataset):
    def __init__(self, demo_path: str, window: int = 16):
        self.demo_path = Path(demo_path)
        self.window = window

        with h5py.File(self.demo_path, "r") as f:
            self.states = f["states"][:]
            self.actions = f["actions"][:]

        assert len(self.states) == len(self.actions)
        assert len(self.states) > self.window

    def __len__(self):
        return len(self.states) - self.window

    def __getitem__(self, idx):
        states = self.states[idx:idx + self.window]
        actions = self.actions[idx:idx + self.window]

        return {
            "states": torch.tensor(states, dtype=torch.float32),
            "actions": torch.tensor(actions, dtype=torch.float32),
        }


def main():
    ds = DummyCarryDataset("examples/demo_000.hdf5", window=16)
    sample = ds[0]
    print("len:", len(ds))
    print("states:", sample["states"].shape)
    print("actions:", sample["actions"].shape)


if __name__ == "__main__":
    main()
