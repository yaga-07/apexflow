# ApexFlow CFD environment
# Base: ESI OpenCFD OpenFOAM v2512 (latest stable as of 2026)
# Multi-arch: native linux/arm64 on Apple Silicon, native linux/amd64 on Intel.
# Run with --user $(id -u):$(id -g) to inherit host user (handled in compose file).

FROM opencfd/openfoam-default:2512

# Install Python 3 and scientific stack for in-container scripting.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-numpy \
    python3-scipy \
    python3-yaml \
    vim \
    less \
    && rm -rf /var/lib/apt/lists/*

# The base image's /etc/bash.bashrc already sources OpenFOAM environment
# system-wide, so it works for any UID.

CMD ["/bin/bash"]
