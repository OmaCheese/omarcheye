#!/bin/bash
# Undo install.sh: stop and remove the systemd user service, the `omarcheye`
# command, the Python environment, the models and the input helper. Your
# calibration (~/.local/state/omarcheye) and settings (~/.config/omarcheye) stay;
# the last line says how to delete them too. Then remove the plugin itself:
# omarchy plugin remove omacheese.omarcheye
set -euo pipefail

data=${XDG_DATA_HOME:-$HOME/.local/share}/omarcheye
cache=${XDG_CACHE_HOME:-$HOME/.cache}/omarcheye
state=${XDG_STATE_HOME:-$HOME/.local/state}/omarcheye
settings=${XDG_CONFIG_HOME:-$HOME/.config}/omarcheye
units=${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user
link=$HOME/.local/bin/omarcheye

say() { printf '==> %s\n' "$*"; }

say "systemd user service"
systemctl --user stop omarcheye.service 2>/dev/null || true
systemctl --user reset-failed omarcheye.service 2>/dev/null || true
if [[ -f $units/omarcheye.service ]]; then
  rm -f "$units/omarcheye.service"
  systemctl --user daemon-reload
fi

say "omarcheye command"
if [[ -L $link && $(readlink "$link") == */bin/omarcheye ]]; then
  rm -f "$link"
fi

say "Python environment, models and input helper"
rm -rf "$data" "$cache"

say "Done. Kept your calibration and settings; to delete them too: rm -rf $state $settings"
