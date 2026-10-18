# Build the C encoder, its command line tool and its unit tests.
#
#   make            library + frbpe_cli
#   make test       build and run the C unit tests
#   make SAN=1 ...  same with AddressSanitizer and UndefinedBehaviorSanitizer
#                   (Linux/macOS; MinGW on Windows does not ship them)

CC ?= gcc
CFLAGS ?= -O2
CFLAGS += -std=c99 -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Werror -Iinclude -Isrc
LDFLAGS ?=

ifeq ($(SAN),1)
CFLAGS += -g -O1 -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=all
LDFLAGS += -fsanitize=address,undefined
endif

ifeq ($(OS),Windows_NT)
EXE := .exe
BENCH_LIBS := -lpsapi
endif

BUILD := build
SRC := src/utf8.c src/pretokenize.c src/bpe.c
OBJ := $(SRC:src/%.c=$(BUILD)/%.o)
LIB := $(BUILD)/libfrbpe.a
CLI := $(BUILD)/frbpe_cli$(EXE)
TEST := $(BUILD)/test_frbpe$(EXE)
BENCH := $(BUILD)/frbpe_bench$(EXE)

.PHONY: all test clean

all: $(CLI) $(BENCH)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/%.o: src/%.c include/frbpe.h src/internal.h src/unicode_tables.h | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

$(LIB): $(OBJ)
	$(AR) rcs $@ $^

$(CLI): tools/frbpe_cli.c $(LIB)
	$(CC) $(CFLAGS) $< $(LIB) $(LDFLAGS) -o $@

$(BENCH): tools/frbpe_bench.c $(LIB)
	$(CC) $(CFLAGS) $< $(LIB) $(LDFLAGS) $(BENCH_LIBS) -o $@

$(TEST): tests/test_frbpe.c $(LIB)
	$(CC) $(CFLAGS) $< $(LIB) $(LDFLAGS) -o $@

test: $(TEST) $(CLI) $(BENCH)
	./$(TEST)

clean:
	rm -rf $(BUILD)
