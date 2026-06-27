import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.dataset import DummyCarryDataset
from models.wam import MinimalWAM


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = DummyCarryDataset("examples/demo_000.hdf5", window=16)
    dl = DataLoader(ds, batch_size=32, shuffle=True)

    sample = ds[0]
    state_dim = sample["states"].shape[-1]
    action_dim = sample["actions"].shape[-1]

    model = MinimalWAM(state_dim=state_dim, action_dim=action_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(3):
        total = 0.0
        for batch in dl:
            states = batch["states"].to(device)
            actions = batch["actions"].to(device)

            # 用当前 state 预测当前 action，只是 Stage 0 smoke test
            pred = model(states[:, 0])
            target = actions[:, 0]

            loss = F.mse_loss(pred, target)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()

        print(f"epoch={epoch} loss={total / len(dl):.6f}")

    print("dummy train finished on", device)


if __name__ == "__main__":
    main()
