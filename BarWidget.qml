import QtQuick
import Quickshell
import Quickshell.Io
import qs.Commons
import qs.Ui

// omarch-eye in the bar: an eye, in the accent colour while omarch-eye moves focus
// to the window you look at. A left click turns it on or off; until omarch-eye is
// set up and calibrated, it opens a terminal that does that first. A right click
// opens a terminal menu: calibrate, refine, test, recentre, preview, camera view.
BarWidget {
  id: root
  moduleName: "omacheese.omarcheye"

  // This plugin's folder, wherever it was installed.
  readonly property string pluginDir: decodeURIComponent(Qt.resolvedUrl(".").toString().replace(/^file:\/\//, "")).replace(/\/$/, "")
  readonly property string cli: pluginDir + "/bin/omarcheye"

  // "setup" (install.sh hasn't run), "calibrate" (no calibration yet), or what
  // systemctl says about the service: active, inactive, activating, failed...
  // Empty until the first check.
  property string status: ""
  readonly property bool running: status === "active" || status === "activating"

  readonly property string tooltip: {
    switch (status) {
    case "":
      return "omarch-eye"
    case "setup":
      return "omarch-eye is not set up yet.\nClick to set it up."
    case "calibrate":
      return "omarch-eye is not calibrated yet.\nClick to calibrate: follow the dots with your eyes for about 30 s."
    case "active":
      return "omarch-eye is on: the window you look at gets focus.\nClick to turn it off. Right-click for calibration, the preview and more."
    case "activating":
      return "omarch-eye is starting…"
    case "failed":
      return "omarch-eye stopped with an error (journalctl --user -u omarcheye).\nClick to start it again."
    default:
      return "omarch-eye is off.\nClick to turn it on. Right-click for calibration, the preview and more."
    }
  }

  function quoted(text) {
    return "'" + String(text).replace(/'/g, "'\\''") + "'"
  }

  function inTerminal(command) {
    Util.execArgv(["omarchy-launch-floating-terminal-with-presentation", command])
  }

  function primaryAction() {
    if (status === "") return
    if (status === "setup") inTerminal(quoted(pluginDir + "/install.sh"))
    else if (status === "calibrate") inTerminal(quoted(cli) + " calibrate")
    else if (!toggleProcess.running) toggleProcess.running = true
  }

  function refresh() {
    if (!checkProcess.running) checkProcess.running = true
  }

  implicitWidth: button.implicitWidth
  implicitHeight: button.implicitHeight

  Process {
    id: checkProcess
    command: ["bash", "-c",
      'data=${XDG_DATA_HOME:-$HOME/.local/share}/omarcheye; '
      + 'state=${XDG_STATE_HOME:-$HOME/.local/state}/omarcheye; '
      + 'units=${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user; '
      + 'if [[ ! -x $data/venv/bin/python || ! -f $units/omarcheye.service ]]; then echo setup; '
      + 'elif [[ ! -f $state/calibration.json ]]; then echo calibrate; '
      + 'else systemctl --user is-active omarcheye.service; fi']
    stdout: StdioCollector {
      waitForEnd: true
      onStreamFinished: root.status = String(text || "").trim()
    }
  }

  Process {
    id: toggleProcess
    command: [root.cli, "toggle"]
    onExited: root.refresh()
  }

  Timer {
    interval: 3000
    running: true
    repeat: true
    triggeredOnStart: true
    onTriggered: root.refresh()
  }

  BarIconButton {
    id: button
    anchors.fill: parent
    bar: root.bar
    // Material Design eye and eye-off from the Nerd Font (U+F0208, U+F0209).
    text: root.running ? "󰈈" : "󰈉"
    active: root.running || root.status === "failed"
    activeColor: root.status === "failed" ? (root.bar ? root.bar.urgent : Color.urgent) : Color.accent
    dimmed: root.status === "setup" || root.status === "calibrate"
    tooltipText: root.tooltip

    onPressed: function (mouseButton) {
      if (mouseButton === Qt.RightButton) root.inTerminal(root.quoted(root.pluginDir + "/bin/omarcheye-menu"))
      else if (mouseButton === Qt.LeftButton) root.primaryAction()
    }
  }
}
