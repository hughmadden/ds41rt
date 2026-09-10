variable "BASE_IMAGE" {
  default = "nvcr.io/nvidia/pytorch:26.05-py3"
}

group "default" {
  targets = ["coordinator", "expert"]
}

target "coordinator" {
  dockerfile = "docker/Dockerfile.dev"
  tags = ["ds41rt-coordinator-dev"]
  args = {
    BASE_IMAGE = BASE_IMAGE
    DS41RT_ROLE = "coordinator"
    CUDA_ARCH = "120"
    TARGET_PLATFORM = "linux/amd64"
  }
  platforms = ["linux/amd64"]
}

target "expert" {
  dockerfile = "docker/Dockerfile.dev"
  tags = ["ds41rt-spark-expert-dev"]
  args = {
    BASE_IMAGE = BASE_IMAGE
    DS41RT_ROLE = "expert"
    CUDA_ARCH = "121"
    TARGET_PLATFORM = "linux/arm64"
  }
  platforms = ["linux/arm64"]
}
