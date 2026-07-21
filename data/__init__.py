"""Dataset contracts, collection adapters, schemas, and storage backends.

The dataset package may consume the public :mod:`envs` runtime contract.  It
must not be imported by :mod:`envs` or :mod:`models`.
"""

from data.trajectory import (
    MULTIMODAL_WAM_SCHEMA_VERSION,
    PROPRIO_WAM_SCHEMA_VERSION,
    FieldSpec,
    TrajectorySchema,
    schema_profile,
)

__all__ = [
    "FieldSpec",
    "MULTIMODAL_WAM_SCHEMA_VERSION",
    "PROPRIO_WAM_SCHEMA_VERSION",
    "TrajectorySchema",
    "schema_profile",
]
