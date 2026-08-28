#!/usr/bin/env bash
set -Eeuo pipefail
cd /workspace/repos/rdt-1b
export RDT_AUDIT_CHECKPOINT=${RDT_AUDIT_CHECKPOINT:?}
export RDT_AUDIT_OUTPUT=${RDT_AUDIT_OUTPUT:?}
exec /workspace/venvs/rdt/bin/python /workspace/repos/before-we-act/deployment/rdt_local/audit_checkpoint.py
