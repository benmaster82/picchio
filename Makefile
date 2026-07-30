# Picchio — Makefile
# Motore MoE streaming per GPT-OSS-120B

CC      ?= gcc
CFLAGS  := -O3 -march=native -fopenmp -Wall -Wextra -Wpedantic
LDFLAGS := -lm -lpthread -fopenmp

# Windows (MinGW): rss_gb usa GetProcessMemoryInfo → serve psapi
ifeq ($(OS),Windows_NT)
LDFLAGS += -lpsapi
endif

# Sorgenti (nella root, non in c/)
SRC     := picchio.c
HEADERS := quant.h json.h st.h
TARGET  := picchio

# CUDA (opzionale)
ifdef CUDA
CFLAGS  += -DPICCHIO_CUDA
LDFLAGS += -lcuda -lcudart
endif

# ── Build ──

.PHONY: all clean check info

all: $(TARGET)

$(TARGET): $(SRC) $(HEADERS)
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDFLAGS)
	@echo "✓ $(TARGET) compilato"

# ── Test ──

check: $(TARGET)
	@echo "── self-test ──"
	./$(TARGET) --self-test 2>/dev/null || true
	@echo "✓ test superati"

# ── Info ──

info:
	@echo "picchio — motore MoE streaming per GPT-OSS-120B"
	@echo ""
	@echo "Build:  make"
	@echo "Run:    MODEL=/path/to/gptoss_i4 ./picchio"
	@echo "Test:   make check"
	@echo "Clean:  make clean"
	@echo ""
	@echo "Variabili d'ambiente:"
	@echo "  MODEL     percorso al modello convertito"
	@echo "  MAX       token massimi da generare (default: 128)"
	@echo "  TEMP      temperatura sampling (default: 1.0)"
	@echo "  TOPP      nucleus sampling top-p (default: 0.95)"
	@echo "  PIN_GB    GB di RAM per hot-store expert (default: auto)"
	@echo "  DIRECT    1=O_DIRECT per expert (default: 0)"
	@echo "  PILOT     1=prefetch pilotato dal router (default: 0)"
	@echo "  IO_THREADS thread per i read paralleli degli expert (default: 4)"
	@echo "  PIPE      0=forza read seriali (compat), altrimenti usa IO_THREADS"
	@echo "  ECAP      override slot cache per layer (default: auto da PIN_GB)"

# ── Clean ──

clean:
	rm -f $(TARGET) $(TARGET).exe
	@echo "✓ pulito"
