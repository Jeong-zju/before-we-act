#!/usr/bin/env bash

# Run an official Hugging Face CLI command with the exact interactive token,
# bounded exponential backoff, and an explicit Xet policy. The token is passed
# only in the child environment and is never appended to argv.
hf_with_retry() {
  if (( $# < 3 )); then
    printf >&2 'usage: hf_with_retry LABEL DISABLE_XET COMMAND [ARG ...]\n'
    return 2
  fi
  local label="$1"
  local disable_xet="$2"
  shift 2

  : "${HF_TOKEN:?HF_TOKEN is required for Hugging Face downloads}"
  local max_attempts="${HF_DOWNLOAD_ATTEMPTS:-5}"
  local delay="${HF_DOWNLOAD_INITIAL_BACKOFF_SECONDS:-15}"
  if [[ ! "${max_attempts}" =~ ^[1-9][0-9]*$ ]]; then
    printf >&2 'HF_DOWNLOAD_ATTEMPTS must be a positive integer.\n'
    return 2
  fi
  if [[ ! "${delay}" =~ ^[0-9]+$ ]]; then
    printf >&2 \
      'HF_DOWNLOAD_INITIAL_BACKOFF_SECONDS must be a non-negative integer.\n'
    return 2
  fi
  if [[ "${disable_xet}" != "0" && "${disable_xet}" != "1" ]]; then
    printf >&2 'DISABLE_XET must be 0 or 1.\n'
    return 2
  fi

  local attempt=1
  while (( attempt <= max_attempts )); do
    printf 'Hugging Face %s: attempt %d/%d (Xet %s).\n' \
      "${label}" \
      "${attempt}" \
      "${max_attempts}" \
      "$([[ "${disable_xet}" == "1" ]] && printf disabled || printf enabled)"
    if HF_TOKEN="${HF_TOKEN}" \
      HF_HUB_DISABLE_XET="${disable_xet}" \
      "$@"; then
      return 0
    fi
    if (( attempt == max_attempts )); then
      printf >&2 \
        'Hugging Face %s failed after %d attempts; completed files remain reusable.\n' \
        "${label}" \
        "${max_attempts}"
      return 1
    fi
    printf >&2 \
      'Hugging Face %s failed; retrying in %d seconds.\n' \
      "${label}" \
      "${delay}"
    sleep "${delay}"
    attempt=$((attempt + 1))
    if (( delay < 300 )); then
      delay=$((delay * 2))
      if (( delay > 300 )); then
        delay=300
      fi
    fi
  done
}

hf_download_with_retry() {
  if (( $# < 3 )); then
    printf >&2 \
      'usage: hf_download_with_retry LABEL DISABLE_XET REPO_ID [ARG ...]\n'
    return 2
  fi
  local label="$1"
  local disable_xet="$2"
  shift 2
  hf_with_retry \
    "${label}" \
    "${disable_xet}" \
    uv run --frozen hf download "$@"
}
