#!/usr/bin/env bash

# These functions intentionally keep the input token alive while replacing a
# stale FIFO. The token is cleared only after a reader has received it.
s0_prepare_hf_token_fifo() {
  : "${HF_TOKEN_FIFO:?set HF_TOKEN_FIFO}"
  unlink "${HF_TOKEN_FIFO}" 2>/dev/null || true
  mkfifo "${HF_TOKEN_FIFO}"
  chmod 600 "${HF_TOKEN_FIFO}"
}

s0_cleanup_hf_secret() {
  HF_TOKEN_INPUT=""
  unset HF_TOKEN_INPUT
  unlink "${HF_TOKEN_FIFO}" 2>/dev/null || true
}

s0_deliver_hf_token() {
  : "${HF_TOKEN_INPUT:?read HF_TOKEN_INPUT before delivering it}"
  printf '%s\n' "${HF_TOKEN_INPUT}" >"${HF_TOKEN_FIFO}"
  s0_cleanup_hf_secret
}
