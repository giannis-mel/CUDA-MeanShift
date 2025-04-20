# 1) Builder stage: compile code with CUDA toolkit on Ubuntu 22.04
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 AS builder

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# Copy source and Makefile
COPY Makefile serial.c nonshared.cu shared.cu input.txt output_reference.txt ./

# Build all binaries
RUN make all

# 2) Runtime stage
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Install make
RUN apt-get update && apt-get install -y --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/serial ./serial
COPY --from=builder /app/nonshared ./nonshared
COPY --from=builder /app/shared ./shared
COPY --from=builder /app/input.txt ./input.txt
COPY --from=builder /app/output_reference.txt ./output_reference.txt
COPY --from=builder /app/Makefile ./Makefile

CMD ["make", "run_all"]
