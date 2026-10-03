#!/bin/bash
# Set omeye up on this machine: Python environment, face model, input helper,
# the `omeye` command in ~/.local/bin and the systemd user service.
# Safe to re-run after a `git pull`.
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
data=${XDG_DATA_HOME:-$HOME/.local/share}/omeye
units=${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user
model_url=https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task

say() { printf '==> %s\n' "$*"; }

missing=()
for pkg in gtk4-layer-shell python-gobject python-cairo wayland-protocols v4l-utils; do
  pacman -Q "$pkg" &>/dev/null || missing+=("$pkg")
done
if ((${#missing[@]})); then
  echo "install.sh: missing system packages: sudo pacman -S ${missing[*]}" >&2
  exit 1
fi

say "Python environment"
uv sync --quiet --project "$root"

say "Face landmark model"
if [[ ! -s $data/models/face_landmarker.task ]]; then
  mkdir -p "$data/models"
  curl -fsSLo "$data/models/face_landmarker.task.part" "$model_url"
  mv "$data/models/face_landmarker.task.part" "$data/models/face_landmarker.task"
fi

say "Input-activity helper"
make -C "$root" --quiet

say "omeye command"
mkdir -p "$HOME/.local/bin"
ln -sfn "$root/bin/omeye" "$HOME/.local/bin/omeye"

say "systemd user service"
mkdir -p "$units"
sed "s|@ROOT@|$root|g" "$root/systemd/omeye.service" >"$units/omeye.service"
systemctl --user daemon-reload

say "Done. Next: omeye calibrate, then omeye on (or your toggle key)"
