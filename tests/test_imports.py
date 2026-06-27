def test_imports():
    import torch
    import mujoco
    import h5py
    import gymnasium
    import hydra
    assert torch is not None
    assert mujoco is not None
