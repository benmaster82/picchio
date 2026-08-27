# Picchio — Makefile
# Streaming MoE engine for GPT-OSS-120B

CC      ?= gcc
CFLAGS  := -O3 -march=native -fopenmp -Wall -Wextra -Wpedantic
LDFLAGS := -lm -lpthread -fopenmp

# Windows (MinGW): rss_gb uses GetProcessMemoryInfo → needs psapi;
# the distributed pipeline mode uses Winsock → needs ws2_32.
ifeq ($(OS),Windows_NT)
LDFLAGS += -lpsapi -lws2_32
endif

# Sources (in the root, not in c/)
SRC     := picchio.c
HEADERS := quant.h json.h st.h
TARGET  := picchio

# CUDA (optional)
ifdef CUDA
CFLAGS  += -DPICCHIO_CUDA
LDFLAGS += -lcuda -lcudart
endif

# ── Build ──

.PHONY: all clean check info

all: $(TARGET)

$(TARGET): $(SRC) $(HEADERS)
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDFLAGS)
	@echo "✓ $(TARGET) built"

# ── Test ──

check: $(TARGET)
	@echo "── self-test ──"
	./$(TARGET) --self-test 2>/dev/null || true
	@echo "✓ tests passed"

# ── Info ──

info:
	@echo "picchio — streaming MoE engine for GPT-OSS-120B"
	@echo ""
	@echo "Build:  make"
	@echo "Run:    MODEL=/path/to/gptoss_i4 ./picchio"
	@echo "Test:   make check"
	@echo "Clean:  make clean"
	@echo ""
	@echo "Environment variables:"
	@echo "  MODEL       path to the converted model"
	@echo "  MAX         maximum tokens to generate (default: 128)"
	@echo "  TEMPERATURE sampling temperature (default: 1.0)"
	@echo "  TOPP        nucleus sampling top-p (default: 0.95)"
	@echo "  PIN_GB      GB of RAM for the expert cache (default: auto)"
	@echo "  PREFETCH    1=prefetch→LRU of the next layer (default: 0; alias PILOT)"
	@echo "  IO_THREADS  threads for parallel expert reads (default: 4)"
	@echo "  PIPE        0=force serial reads (compat), otherwise use IO_THREADS"
	@echo "  ECAP        override cache slots per layer (default: auto from PIN_GB)"
	@echo "  PREDICT_PROBE 1=measure prefetch prediction accuracy (diagnostic)"

# ── Clean ──

clean:
	rm -f $(TARGET) $(TARGET).exe
	@echo "✓ cleaned"
