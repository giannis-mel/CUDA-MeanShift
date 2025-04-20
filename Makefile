# Compiler configurations
CC        := gcc
NVCC      := nvcc

# CUDA architectures: compile for common GPUs and include PTX fallback
CUDA_ARCH_FLAGS := \
  -gencode arch=compute_61,code=sm_61 \
  -gencode arch=compute_75,code=sm_75 \
  -gencode arch=compute_86,code=sm_86 \
  -gencode arch=compute_86,code=compute_86

CFLAGS    := -lm
NVCCFLAGS := -O3 -lm $(CUDA_ARCH_FLAGS)

# Binaries
SERIAL    := serial
NONSHARED := nonshared
SHARED    := shared

# Default convergence input
INPUT     ?= 0.5

# Build all binaries
all: $(SERIAL) $(NONSHARED) $(SHARED)

$(SERIAL): serial.c
	$(CC) $< -o $@ $(CFLAGS)

$(NONSHARED): nonshared.cu
	$(NVCC) $< -o $@ $(NVCCFLAGS)

$(SHARED): shared.cu
	$(NVCC) $< -o $@ $(NVCCFLAGS)

# Run all implementations (requires that binaries are already built)
run_all: run_serial run_nonshared run_shared

run_serial:
	./$(SERIAL) $(INPUT)

run_nonshared:
	./$(NONSHARED) $(INPUT)

run_shared:
	./$(SHARED) $(INPUT)

# Clean generated binaries
clean:
	rm -f $(SERIAL) $(NONSHARED) $(SHARED)

.PHONY: all run_all run_serial run_nonshared run_shared clean
