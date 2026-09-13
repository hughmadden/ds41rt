#include "ds41rt_native.h"
#include <cuda_runtime_api.h>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>

static void check(bool condition) {
  if (!condition) std::abort();
}
struct Gate { std::atomic<bool> release{false}; };
static void CUDART_CB hold(void* state) {
  auto& gate = *static_cast<Gate*>(state);
  while (!gate.release.load(std::memory_order_acquire)) std::this_thread::yield();
}
int main() {
  void* pending = nullptr;
  void* peer = nullptr;
  check(ds41rt_cuda_stream_create(&pending) == DS41RT_STATUS_OK);
  check(ds41rt_cuda_stream_create(&peer) == DS41RT_STATUS_OK);
  check(ds41rt_cuda_stream_query(pending, nullptr) == DS41RT_STATUS_INVALID_ARGUMENT);
  int32_t ready = -1;
  check(ds41rt_cuda_stream_query(pending, &ready) == DS41RT_STATUS_OK && ready == 1);
  Gate gate;
  check(cudaLaunchHostFunc(static_cast<cudaStream_t>(pending), hold, &gate) == cudaSuccess);
  // The gate cannot complete until after these queries. A blocking implementation
  // hits the CTest deadline instead of silently passing as a completion query.
  check(ds41rt_cuda_stream_query(pending, &ready) == DS41RT_STATUS_OK && ready == 0);
  check(ds41rt_cuda_stream_query(peer, &ready) == DS41RT_STATUS_OK && ready == 1);
  gate.release.store(true, std::memory_order_release);
  check(ds41rt_cuda_stream_synchronize(pending) == DS41RT_STATUS_OK);
  check(ds41rt_cuda_stream_query(pending, &ready) == DS41RT_STATUS_OK && ready == 1);
  check(ds41rt_cuda_stream_destroy(peer) == DS41RT_STATUS_OK);
  check(ds41rt_cuda_stream_destroy(pending) == DS41RT_STATUS_OK);
  std::cout << "stream query distinguishes pending and complete without joining the peer\n";
}
