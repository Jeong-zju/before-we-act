import argparse
from pathlib import Path

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", type=str, required=True)
    args = parser.parse_args()

    demo_path = Path(args.demo)
    if not demo_path.exists():
        raise FileNotFoundError(demo_path)

    with h5py.File(demo_path, "r") as f:
        states = f["states"][:]
        actions = f["actions"][:]
        contacts = f["contacts"][:]
        times = f["times"][:]

    print("Loaded demo:", demo_path)
    print("states:", states.shape)
    print("actions:", actions.shape)
    print("contacts:", contacts.shape)
    print("times:", times.shape)

    print("\\nFirst step:")
    print("time:", times[0])
    print("state[:10]:", np.round(states[0, :10], 4))
    print("action:", np.round(actions[0], 4))
    print("contacts:", contacts[0])

    print("\\nLast step:")
    print("time:", times[-1])
    print("state[:10]:", np.round(states[-1, :10], 4))
    print("action:", np.round(actions[-1], 4))
    print("contacts:", contacts[-1])


if __name__ == "__main__":
    main()
