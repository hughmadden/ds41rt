variable "BASE_IMAGE" {
  default = "nvcr.io/nvidia/pytorch:26.05-py3"
}

variable "QUANT_COORDINATOR_BASE_IMAGE" {
  # Native amd64 child of the NGC 26.05 multi-platform manifest.
  default = "nvcr.io/nvidia/pytorch@sha256:ca73b4795f0d3ae27e9cd81b1b1f1b7fc6c0a129f7d51a359d2326e95af48a3d"
}

variable "QUANT_EXPERT_BASE_IMAGE" {
  # Native arm64 child of the same NGC 26.05 multi-platform manifest.
  default = "nvcr.io/nvidia/pytorch@sha256:aa400d4373fa71f30e1714664beabcc64c2d198e72d65e9a3641440b07e7cc83"
}

group "default" {
  targets = ["coordinator", "expert"]
}

group "quantization" {
  targets = ["quant-coordinator", "quant-expert"]
}

target "coordinator" {
  dockerfile = "docker/Dockerfile.dev"
  tags = ["ds4rt-coordinator-dev"]
  args = {
    BASE_IMAGE = BASE_IMAGE
    DS4RT_ROLE = "coordinator"
    CUDA_ARCH = "120"
    TARGET_PLATFORM = "linux/amd64"
  }
  platforms = ["linux/amd64"]
}

target "expert" {
  dockerfile = "docker/Dockerfile.dev"
  tags = ["ds4rt-spark-expert-dev"]
  args = {
    BASE_IMAGE = BASE_IMAGE
    DS4RT_ROLE = "expert"
    CUDA_ARCH = "121"
    TARGET_PLATFORM = "linux/arm64"
  }
  platforms = ["linux/arm64"]
}

# Build each quantization target on its native machine. In particular, invoke
# quant-expert from a Spark builder; the target platform is a contract, not an
# invitation to emulate arm64 on the coordinator.
target "quant-coordinator" {
  dockerfile = "docker/Dockerfile.quantization"
  tags = ["ds4rt-quant-coordinator"]
  args = {
    BASE_IMAGE = QUANT_COORDINATOR_BASE_IMAGE
    DS4RT_QUANT_ROLE = "coordinator"
    DS4RT_QUANT_TARGET_PLATFORM = "linux/amd64"
    DS4RT_QUANT_CUDA_ARCH = "120"
    DS4RT_QUANT_MIN_GPUS = "2"
    DS4RT_GPTQMODEL_COMMIT = "5624c3d716b63218d18fd662eb4326f14f4beff0"
    DS4RT_QUANT_REQUIREMENTS_SHA256 = "0dd72f6171711269548da684cd1f4fda37d0da2948f371129f52fcae0bdac140"
    DS4RT_QUANT_BUILD_REQUIREMENTS_SHA256 = "9f21166fd088fd5eee2e9560c5d97b14201e0fde30d7ee27a43a56c24e104fd1"
  }
  platforms = ["linux/amd64"]
}

target "quant-expert" {
  dockerfile = "docker/Dockerfile.quantization"
  tags = ["ds4rt-quant-spark-expert"]
  args = {
    BASE_IMAGE = QUANT_EXPERT_BASE_IMAGE
    DS4RT_QUANT_ROLE = "expert"
    DS4RT_QUANT_TARGET_PLATFORM = "linux/arm64"
    DS4RT_QUANT_CUDA_ARCH = "121"
    DS4RT_QUANT_MIN_GPUS = "1"
    DS4RT_GPTQMODEL_COMMIT = "5624c3d716b63218d18fd662eb4326f14f4beff0"
    DS4RT_QUANT_REQUIREMENTS_SHA256 = "196cfaf026ef3f7d9b81d65aaeb00a871ad69793381a40ce0d1b3decbcc31d7b"
    DS4RT_QUANT_BUILD_REQUIREMENTS_SHA256 = "9f21166fd088fd5eee2e9560c5d97b14201e0fde30d7ee27a43a56c24e104fd1"
  }
  platforms = ["linux/arm64"]
}
