// omeye-idle: print "active" when keyboard or mouse input arrives and "idle"
// after <timeout_ms> without input. omeye reads this on stdout to pause gaze
// focus while you type. Uses ext-idle-notify-v1 (no access to /dev/input
// needed); version 2's input-idle request ignores idle inhibitors, so a
// playing video does not make every moment look like typing.
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <wayland-client.h>
#include "ext-idle-notify-v1-client-protocol.h"

static struct wl_seat *seat;
static struct ext_idle_notifier_v1 *notifier;
static uint32_t notifier_version;

static void registry_global(void *data, struct wl_registry *registry, uint32_t name,
                            const char *interface, uint32_t version) {
	if (strcmp(interface, wl_seat_interface.name) == 0 && !seat) {
		seat = wl_registry_bind(registry, name, &wl_seat_interface, 1);
	} else if (strcmp(interface, ext_idle_notifier_v1_interface.name) == 0) {
		notifier_version = version < 2 ? version : 2;
		notifier = wl_registry_bind(registry, name, &ext_idle_notifier_v1_interface, notifier_version);
	}
}

static void registry_global_remove(void *data, struct wl_registry *registry, uint32_t name) {}

static const struct wl_registry_listener registry_listener = {
	.global = registry_global,
	.global_remove = registry_global_remove,
};

static void say(const char *state) {
	if (puts(state) == EOF || fflush(stdout) == EOF)
		exit(0);
}

static void idled(void *data, struct ext_idle_notification_v1 *n) { say("idle"); }
static void resumed(void *data, struct ext_idle_notification_v1 *n) { say("active"); }

static const struct ext_idle_notification_v1_listener notification_listener = {
	.idled = idled,
	.resumed = resumed,
};

int main(int argc, char **argv) {
	long timeout_ms = argc > 1 ? strtol(argv[1], NULL, 10) : 700;
	if (timeout_ms <= 0) {
		fprintf(stderr, "usage: omeye-idle [timeout_ms]\n");
		return 2;
	}
	prctl(PR_SET_PDEATHSIG, SIGTERM);

	struct wl_display *display = wl_display_connect(NULL);
	if (!display) {
		fprintf(stderr, "omeye-idle: cannot connect to the Wayland display\n");
		return 1;
	}
	struct wl_registry *registry = wl_display_get_registry(display);
	wl_registry_add_listener(registry, &registry_listener, NULL);
	wl_display_roundtrip(display);
	if (!seat || !notifier) {
		fprintf(stderr, "omeye-idle: compositor has no ext-idle-notify-v1\n");
		return 1;
	}

	struct ext_idle_notification_v1 *notification = notifier_version >= 2
		? ext_idle_notifier_v1_get_input_idle_notification(notifier, timeout_ms, seat)
		: ext_idle_notifier_v1_get_idle_notification(notifier, timeout_ms, seat);
	ext_idle_notification_v1_add_listener(notification, &notification_listener, NULL);
	fprintf(stderr, "omeye-idle: protocol v%u (%s), %ld ms\n", notifier_version,
	        notifier_version >= 2 ? "input idle" : "plain idle", timeout_ms);
	say("active");

	while (wl_display_dispatch(display) != -1) {
	}
	return 0;
}
