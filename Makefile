# Makefile to compile and run programs

# Compiler and linker configurations
CC=gcc
NVCC=nvcc
CFLAGS=-lm
NVCCFLAGS=-O3 -lm

# Target binaries
SERIAL=serial
NONSHARED=nonshared
SHARED=shared

# Input argument for running the binaries, default value
INPUT?=0.5

# Default target
all: compile_all

# Compile all programs
compile_all: $(SERIAL) $(NONSHARED) $(SHARED)

# Compile serial.c
$(SERIAL): serial.c
	$(CC) serial.c -o $(SERIAL) $(CFLAGS)

# Compile nonshared.cu
$(NONSHARED): nonshared.cu
	$(NVCC) nonshared.cu -o $(NONSHARED) $(NVCCFLAGS)

# Compile shared.cu
$(SHARED): shared.cu
	$(NVCC) shared.cu -o $(SHARED) $(NVCCFLAGS)

# Run all programs
run_all: run_serial run_nonshared run_shared

# Run serial
run_serial: $(SERIAL)
	./$(SERIAL) $(INPUT)

# Run nonshared
run_nonshared: $(NONSHARED)
	./$(NONSHARED) $(INPUT)

# Run shared
run_shared: $(SHARED)
	./$(SHARED) $(INPUT)

# Clean up
clean:
	rm -f $(SERIAL) $(NONSHARED) $(SHARED)

# PHONY targets
.PHONY: all clean run_all run_serial run_nonshared run_shared compile_all
