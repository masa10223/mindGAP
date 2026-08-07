# shellcheck shell=bash
# mindGAP / ELA_analysis 用デフォルト (source 専用)
NAME="${NAME:-mindGAP}"
PORT="${PORT:-8881}"
IMAGE="${IMAGE:-mindgap:latest}"
REQ_FILE="${REQ_FILE:-requirements-mindgap.txt}"
LOCK_FILE="${LOCK_FILE:-requirements-mindgap.lock}"
