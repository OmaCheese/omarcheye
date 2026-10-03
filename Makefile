# Builds the input-activity helper. `make` -> build/omeye-idle
PROTOCOLS := $(shell pkg-config --variable=pkgdatadir wayland-protocols)
IDLE_XML := $(PROTOCOLS)/staging/ext-idle-notify/ext-idle-notify-v1.xml
WAYLAND := $(shell pkg-config --cflags --libs wayland-client)

build/omeye-idle: src/omeye-idle.c build/ext-idle-notify-v1-protocol.c build/ext-idle-notify-v1-client-protocol.h
	$(CC) -O2 -Wall -Wextra -Wno-unused-parameter -Ibuild -o $@ src/omeye-idle.c build/ext-idle-notify-v1-protocol.c $(WAYLAND)

build/ext-idle-notify-v1-client-protocol.h: $(IDLE_XML) | build
	wayland-scanner client-header $< $@

build/ext-idle-notify-v1-protocol.c: $(IDLE_XML) | build
	wayland-scanner private-code $< $@

build:
	mkdir -p build

clean:
	rm -rf build

.PHONY: clean
