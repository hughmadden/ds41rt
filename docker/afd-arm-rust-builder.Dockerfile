# ARM64 builder only. Before build, bind the local base tag to verified image
# sha256:622398bf2562f3d570fa8821715b501bfe69be718a00dc0dcf7304420fe4287b.
# Use --pull=false; the Rust stage is pinned to its arm64 manifest.
# Build packages offline from unchanged Cargo.lock, with source mounted read-only.
# ---- stage 1: pinned official arm64 Rust toolchain (only source of cargo/rustc) ----
FROM --platform=linux/arm64 \
     rust:1.98.1-bookworm@sha256:d6eafdebc66e9a7fd9eaf2c5e84a29febf3ab3111a1ca8d2a6ce3916ed10171a \
     AS rust-toolchain

# ---- stage 2: retained Spark runtime; adds only static toolchain files ----
FROM --platform=linux/arm64 ds41rt-spark-base:622398bf2562f3d570fa8821715b501bfe69be718a00dc0dcf7304420fe4287b AS runtime

# Copy the complete rustup home (toolchains/1.98.1-aarch64-unknown-linux-gnu,
# settings.toml, update-hashes) and the rustup proxy bin dir. Nothing else.
COPY --from=rust-toolchain /usr/local/rustup   /usr/local/rustup
COPY --from=rust-toolchain /usr/local/cargo/bin /usr/local/cargo/bin

# Writable Cargo cache is mounted separately; registry only, never credentials.
RUN mkdir -p /cargo

ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/cargo \
    PYO3_PYTHON=/usr/bin/python3 \
    PATH=/usr/local/cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
