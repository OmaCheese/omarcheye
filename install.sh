#!/bin/bash
# Set omarcheye up on this machine: Python environment, face and eye models,
# input helper, the `omarcheye` command in ~/.local/bin and the systemd user
# service. Safe to re-run after an update (`omarchy plugin update` or `git pull`).
#
# Everything it makes lives outside this folder, in ~/.local/share/omarcheye
# (and the command link and unit file): Omarchy reloads its plugins on any change
# inside a plugin's folder. uninstall.sh removes it all again.
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
data=${XDG_DATA_HOME:-$HOME/.local/share}/omarcheye
units=${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user

# Downloaded models, pinned to one version each and checked against its sha256.
face_url=https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
face_sha=64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff
eyenet_url=https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002
eyenet_xml_sha=4f413ad706a622a8f1411e3eec6f40e4d51f062a6eb76c22498970235195ffa0
eyenet_bin_sha=4947253671d1984f0c0b03392854b3af022bfca3b3dd055768db4531f031e1d4

say() { printf '==> %s\n' "$*"; }

# fetch URL SHA256 FILE: download FILE unless it is already there with that checksum.
fetch() {
  if [[ -s $3 ]] && sha256sum --quiet --check <<<"$2  $3" &>/dev/null; then
    return
  fi
  curl -fsSLo "$3.part" "$1"
  if ! sha256sum --quiet --check <<<"$2  $3.part" &>/dev/null; then
    rm -f "$3.part"
    echo "install.sh: $1 did not match its checksum; not installed" >&2
    exit 1
  fi
  mv "$3.part" "$3"
}

missing=()
for pkg in gtk4-layer-shell python-gobject python-cairo wayland-protocols v4l-utils; do
  pacman -Q "$pkg" &>/dev/null || missing+=("$pkg")
done
command -v uv &>/dev/null || missing+=(uv)
for tool in make cc pkg-config; do
  if ! command -v "$tool" &>/dev/null; then
    missing+=(base-devel)
    break
  fi
done
if ((${#missing[@]})); then
  echo "omarcheye needs these packages: ${missing[*]}"
  answer=n
  if [[ -t 0 ]] && command -v omarchy &>/dev/null; then
    read -rp "Install them now with omarchy pkg add (asks for your password)? [y/N] " answer
  fi
  if [[ $answer != [yY]* ]]; then
    echo "install.sh: install them with: omarchy pkg add ${missing[*]}" >&2
    exit 1
  fi
  omarchy pkg add "${missing[@]}"
fi

say "Python environment (in $data/venv)"
UV_PROJECT_ENVIRONMENT=$data/venv uv sync --locked --quiet --project "$root"

say "Face landmark model (Google MediaPipe Face Landmarker, Apache-2.0)"
mkdir -p "$data/models"
fetch "$face_url" "$face_sha" "$data/models/face_landmarker.task"

say "Eye network (Intel Open Model Zoo gaze-estimation-adas-0002, Apache-2.0)"
fetch "$eyenet_url.xml" "$eyenet_xml_sha" "$data/models/gaze-estimation-adas-0002.xml"
fetch "$eyenet_url.bin" "$eyenet_bin_sha" "$data/models/gaze-estimation-adas-0002.bin"

say "Input-activity helper"
make -C "$root" --quiet BUILD="$data/build"

say "omarcheye command"
mkdir -p "$HOME/.local/bin"
ln -sfn "$root/bin/omarcheye" "$HOME/.local/bin/omarcheye"

say "systemd user service"
mkdir -p "$units"
sed "s|@ROOT@|$root|g" "$root/systemd/omarcheye.service" >"$units/omarcheye.service"
systemctl --user daemon-reload

say "Done. Next: omarcheye calibrate, then omarcheye on (or click the eye in the bar)"
