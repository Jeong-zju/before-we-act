from pathlib import Path
import h5py
import numpy as np

from envs.mujoco_carry_env import MinimalTwoRobotCarryEnv


def main():
    out_dir = Path("examples")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "demo_000.hdf5"

    env = MinimalTwoRobotCarryEnv()
    obs = env.reset()

    states = []
    actions = []
    contacts = []
    times = []

    done = False
    while not done:
        action = env.scripted_action()
        obs, reward, done, info = env.step(action)

        states.append(env.get_state_vector())
        actions.append(action.copy())
        contacts.append(info["ncon"])
        times.append(obs["time"])

    states = np.asarray(states)
    actions = np.asarray(actions)
    contacts = np.asarray(contacts)
    times = np.asarray(times)

    with h5py.File(out_path, "w") as f:
        f.create_dataset("states", data=states)
        f.create_dataset("actions", data=actions)
        f.create_dataset("contacts", data=contacts)
        f.create_dataset("times", data=times)

    print(f"saved: {out_path}")
    print("states:", states.shape)
    print("actions:", actions.shape)
    print("contacts:", contacts.shape)


if __name__ == "__main__":
    main()
