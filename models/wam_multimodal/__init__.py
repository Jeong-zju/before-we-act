"""Visual-conditioned latent WAM modules for multimodal phases M1 and M2."""

from models.wam_multimodal.agent_factorized_flow_wam import (
    AgentFactorizedFlowWAM,
)
from models.wam_multimodal.block_causal_transformer import (
    BlockCausalWAM,
    BlockCausalWAMConfig,
    BlockCausalWAMOutput,
    UTF8TaskEncoder,
    build_block_causal_attention_mask,
)

from models.wam_multimodal.latent_wam import (
    CapacityControl,
    LatentWAM,
    LatentWAMConfig,
    LatentWAMEncoding,
    LatentWAMOutput,
)
from models.wam_multimodal.latent_world_head import (
    ActionConditionedFutureLatentHead,
    CANONICAL_FUTURE_HORIZONS,
    FutureLatentHeadConfig,
)
from models.wam_multimodal.local_future_predictor import (
    LocalActionConditionedFuturePredictor,
    LocalFuturePredictorConfig,
)
from models.wam_multimodal.protected_hybrid_future_predictor import (
    ProtectedHybridFuturePrediction,
    ProtectedHybridFuturePredictor,
    exact_own_difference,
)
from models.wam_multimodal.protected_team_future_predictor import (
    ProtectedTeamFuturePredictor,
    ProtectedTeamFuturePredictorConfig,
)
from models.wam_multimodal.token_resampler import (
    PerceiverResampler,
    PerceiverResamplerConfig,
    SpatialVisualTokenAdapter,
    VisualAdapterOutput,
)
from models.wam_multimodal.team_shared_future_predictor import (
    TeamSharedFuturePrediction,
    TeamSharedFuturePredictor,
    TeamSharedFuturePredictorConfig,
)
from models.wam_multimodal.vision_encoder import (
    DEFAULT_DINOV3_ENCODER,
    DEFAULT_DINOV3_CONFIG_SHA256,
    DEFAULT_DINOV3_MODEL_ID,
    DEFAULT_DINOV3_REVISION,
    DEFAULT_DINOV3_WEIGHTS_SHA256,
    DINOV3_ENCODER_SPECS,
    DINOV3_PREPROCESS_ID,
    DINOV3_RECTANGULAR_PREPROCESS_ID,
    DINOv3EncoderSpec,
    FrozenDINOv3Config,
    FrozenDINOv3Encoder,
    FrozenResNet18Config,
    FrozenResNet18Encoder,
    IMAGENET_RGB_MEAN,
    IMAGENET_RGB_STD,
    OFFICIAL_RESNET18_FILENAME,
    OFFICIAL_RESNET18_SHA256,
    VisionEncoderOutput,
    build_resnet18_classifier,
    canonical_json_sha256,
    default_resnet18_weights_path,
    sha256_file,
)

__all__ = [
    "ActionConditionedFutureLatentHead",
    "AgentFactorizedFlowWAM",
    "BlockCausalWAM",
    "BlockCausalWAMConfig",
    "BlockCausalWAMOutput",
    "CANONICAL_FUTURE_HORIZONS",
    "CapacityControl",
    "DEFAULT_DINOV3_ENCODER",
    "DEFAULT_DINOV3_CONFIG_SHA256",
    "DEFAULT_DINOV3_MODEL_ID",
    "DEFAULT_DINOV3_REVISION",
    "DEFAULT_DINOV3_WEIGHTS_SHA256",
    "DINOV3_ENCODER_SPECS",
    "DINOV3_PREPROCESS_ID",
    "DINOV3_RECTANGULAR_PREPROCESS_ID",
    "DINOv3EncoderSpec",
    "FrozenDINOv3Config",
    "FrozenDINOv3Encoder",
    "FrozenResNet18Config",
    "FrozenResNet18Encoder",
    "FutureLatentHeadConfig",
    "IMAGENET_RGB_MEAN",
    "IMAGENET_RGB_STD",
    "LatentWAM",
    "LatentWAMConfig",
    "LatentWAMEncoding",
    "LatentWAMOutput",
    "LocalActionConditionedFuturePredictor",
    "LocalFuturePredictorConfig",
    "OFFICIAL_RESNET18_FILENAME",
    "OFFICIAL_RESNET18_SHA256",
    "PerceiverResampler",
    "PerceiverResamplerConfig",
    "ProtectedHybridFuturePrediction",
    "ProtectedHybridFuturePredictor",
    "ProtectedTeamFuturePredictor",
    "ProtectedTeamFuturePredictorConfig",
    "SpatialVisualTokenAdapter",
    "TeamSharedFuturePrediction",
    "TeamSharedFuturePredictor",
    "TeamSharedFuturePredictorConfig",
    "UTF8TaskEncoder",
    "VisualAdapterOutput",
    "VisionEncoderOutput",
    "build_resnet18_classifier",
    "build_block_causal_attention_mask",
    "canonical_json_sha256",
    "default_resnet18_weights_path",
    "exact_own_difference",
    "sha256_file",
]
