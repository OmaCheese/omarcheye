# Builds the input-activity helper. `make` -> ~/.local/share/omarcheye/build/omarcheye-idle
# (outside the checkout: Omarchy reloads its plugins on any change inside a plugin's folder).
PROTOCOLS := $(shell pkg-config --variable=pkgdatadir wayland-protocols)
IDLE_XML := $(PROTOCOLS)/staging/ext-idle-notify/ext-idle-notify-v1.xml
WAYLAND := $(shell pkg-config --cflags --libs wayland-client)
BUILD ?= $(or $(XDG_DATA_HOME),$(HOME)/.local/share)/omarcheye/build

$(BUILD)/omarcheye-idle: src/omarcheye-idle.c $(BUILD)/ext-idle-notify-v1-protocol.c $(BUILD)/ext-idle-notify-v1-client-protocol.h
	$(CC) -O2 -Wall -Wextra -Wno-unused-parameter -I$(BUILD) -o $@ src/omarcheye-idle.c $(BUILD)/ext-idle-notify-v1-protocol.c $(WAYLAND)

$(BUILD)/ext-idle-notify-v1-client-protocol.h: $(IDLE_XML) | $(BUILD)
	wayland-scanner client-header $< $@

$(BUILD)/ext-idle-notify-v1-protocol.c: $(IDLE_XML) | $(BUILD)
	wayland-scanner private-code $< $@

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD)

.PHONY: clean
