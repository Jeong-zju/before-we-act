"""Training-time datasets and optimization utilities."""

from train.multimodal_trajectory_dataset import (
    MultimodalEpisodeRecord,
    MultimodalSequenceDataset,
    MultimodalSequenceIndex,
)
from train.generic_m1_trajectory_dataset import (
    GenericM1ManifestIndex,
    GenericM1WindowDataset,
)
from train.m1_data_protocol import (
    build_m1_window_dataset,
    detect_m1_data_protocol,
    load_m1_data_manifest,
    m1_data_capabilities,
)
from train.m1_scratch_builder import (
    ScratchActionFlowConfig,
    ScratchM1BuildConfig,
    ScratchM1Bundle,
    build_scratch_m1,
)
from train.m1_scratch_checkpointing import (
    load_scratch_m1_checkpoint,
    save_scratch_m1_checkpoint,
)
from train.m1_scratch_training import (
    ScratchM1StageConfig,
    build_scratch_optimizer,
    train_scratch_m1_stage,
    validate_scratch_stage_order,
)
from train.trajectory_dataset import (
    EpisodeSequenceIndex,
    ProprioSequenceDataset,
    discover_episode_paths,
    split_episode_paths,
)

__all__ = [
    "EpisodeSequenceIndex",
    "GenericM1ManifestIndex",
    "GenericM1WindowDataset",
    "MultimodalEpisodeRecord",
    "MultimodalSequenceDataset",
    "MultimodalSequenceIndex",
    "ProprioSequenceDataset",
    "ScratchActionFlowConfig",
    "ScratchM1BuildConfig",
    "ScratchM1Bundle",
    "ScratchM1StageConfig",
    "build_scratch_m1",
    "build_scratch_optimizer",
    "build_m1_window_dataset",
    "detect_m1_data_protocol",
    "discover_episode_paths",
    "load_m1_data_manifest",
    "load_scratch_m1_checkpoint",
    "m1_data_capabilities",
    "split_episode_paths",
    "save_scratch_m1_checkpoint",
    "train_scratch_m1_stage",
    "validate_scratch_stage_order",
]
