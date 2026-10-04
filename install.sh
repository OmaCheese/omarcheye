#!/bin/bash
# Set omarcheye up on this machine: Python environment, face model, input helper,
# the `omarcheye` command in ~/.local/bin and the systemd user service.
# Safe to re-run after a `git pull`.
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
data=${XDG_DATA_HOME:-$HOME/.local/share}/omarcheye
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

say "Eye network (Intel Open Model Zoo gaze-estimation-adas-0002, Apache-2.0)"
eyenet_url=https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002
for ext in xml bin; do
  if [[ ! -s $data/models/gaze-estimation-adas-0002.$ext ]]; then
    curl -fsSLo "$data/models/gaze-estimation-adas-0002.$ext.part" "$eyenet_url.$ext"
    mv "$data/models/gaze-estimation-adas-0002.$ext.part" "$data/models/gaze-estimation-adas-0002.$ext"
  fi
done

say "Input-activity helper"
make -C "$root" --quiet

say "omarcheye command"
mkdir -p "$HOME/.local/bin"
ln -sfn "$root/bin/omarcheye" "$HOME/.local/bin/omarcheye"

say "systemd user service"
mkdir -p "$units"
sed "s|@ROOT@|$root|g" "$root/systemd/omarcheye.service" >"$units/omarcheye.service"
systemctl --user daemon-reload

say "Done. Next: omarcheye calibrate, then omarcheye on (or your toggle key)"
