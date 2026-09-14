#include "ds41rt_native.h"
#include <cuda_runtime_api.h>
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <thread>
#include <vector>

#define check(condition) do { if (!(condition)) { \
  char error[1024]{}; ds41rt_last_error(error, sizeof(error)); \
  std::cerr << "check failed at " << __LINE__ << ": " << #condition << '\n'; \
  std::cerr << error << '\n'; \
  std::abort(); \
} } while (false)
struct Gate { std::atomic<bool> release{false}; };
static void CUDART_CB hold(void* state) {
  auto& gate = *static_cast<Gate*>(state);
  while (!gate.release.load(std::memory_order_acquire)) std::this_thread::yield();
}

int main() {
  int count = 0;
  check(cudaGetDeviceCount(&count) == cudaSuccess);
  if (count < 2) { std::cout << "requires two CUDA devices\n"; return 77; }
  check(ds41rt_cuda_get_device(nullptr) == DS41RT_STATUS_INVALID_ARGUMENT);
  check(ds41rt_cuda_set_device(-1) == DS41RT_STATUS_INVALID_ARGUMENT);
  constexpr size_t bytes = 1024 * 1024;
  for (int source = 0; source < 2; ++source) {
    const int destination = 1 - source;
    ds41rt_device_buffer_t src{}, dst{};
    void *producer = nullptr, *consumer = nullptr, *pending = nullptr, *event = nullptr;
    check(ds41rt_cuda_set_device(source) == DS41RT_STATUS_OK);
    check(ds41rt_cuda_enable_peer(destination) == DS41RT_STATUS_OK);
    check(ds41rt_cuda_enable_peer(destination) == DS41RT_STATUS_OK);
    check(ds41rt_alloc_device_buffer(bytes, &src) == DS41RT_STATUS_OK);
    check(src.device_id == source);
    check(ds41rt_cuda_stream_create(&producer) == DS41RT_STATUS_OK);
    check(ds41rt_cuda_event_create(&event) == DS41RT_STATUS_OK);
    check(ds41rt_cuda_set_device(destination) == DS41RT_STATUS_OK);
    check(ds41rt_cuda_enable_peer(source) == DS41RT_STATUS_OK);
    check(ds41rt_alloc_device_buffer(bytes, &dst) == DS41RT_STATUS_OK);
    check(dst.device_id == destination);
    check(ds41rt_cuda_stream_create(&consumer) == DS41RT_STATUS_OK);
    check(ds41rt_cuda_stream_create(&pending) == DS41RT_STATUS_OK);
    check(ds41rt_copy_peer_async(dst, src, bytes + 1, consumer) == DS41RT_STATUS_INVALID_ARGUMENT);
    check(ds41rt_copy_peer_async(dst, src, bytes, nullptr) == DS41RT_STATUS_INVALID_ARGUMENT);
    std::vector<unsigned char> host(bytes);
    for (int pattern : {37, 191, 0}) {
      check(ds41rt_cuda_set_device(source) == DS41RT_STATUS_OK);
      check(ds41rt_copy_peer_async(dst, src, bytes, consumer) == DS41RT_STATUS_INVALID_ARGUMENT);
      check(cudaMemsetAsync(src.ptr, pattern, bytes, static_cast<cudaStream_t>(producer)) == cudaSuccess);
      check(ds41rt_cuda_event_record(event, producer) == DS41RT_STATUS_OK);
      check(ds41rt_cuda_set_device(destination) == DS41RT_STATUS_OK);
      Gate gate;
      check(cudaLaunchHostFunc(static_cast<cudaStream_t>(pending), hold, &gate) == cudaSuccess);
      check(ds41rt_cuda_stream_wait_event(consumer, event) == DS41RT_STATUS_OK);
      check(ds41rt_copy_peer_async(dst, src, bytes, consumer) == DS41RT_STATUS_OK);
      int32_t ready = 0;
      while (!ready) {
        check(ds41rt_cuda_stream_query(consumer, &ready) == DS41RT_STATUS_OK);
        std::this_thread::yield();
      }
      check(ds41rt_cuda_stream_query(pending, &ready) == DS41RT_STATUS_OK && ready == 0);
      gate.release.store(true, std::memory_order_release);
      check(ds41rt_cuda_stream_synchronize(pending) == DS41RT_STATUS_OK);
      check(ds41rt_copy_d2h(host.data(), dst, bytes) == DS41RT_STATUS_OK);
      for (auto value : host) check(value == pattern);
    }
    ds41rt_device_buffer_t local{};
    check(ds41rt_alloc_device_buffer(bytes, &local) == DS41RT_STATUS_OK);
    std::vector<unsigned char> values(bytes);
    for (size_t i = 0; i < bytes; ++i) values[i] = (i * 17 + 11) % 251;
    check(ds41rt_copy_h2d(local, values.data(), bytes) == DS41RT_STATUS_OK);
    check(ds41rt_cuda_set_device(source) == DS41RT_STATUS_OK);
    check(ds41rt_copy_h2d(src, values.data(), bytes) == DS41RT_STATUS_OK);
    check(ds41rt_copy_device_rows_async(dst, src, 17, 19, 43, 31, consumer) == DS41RT_STATUS_INVALID_ARGUMENT);
    check(ds41rt_cuda_set_device(destination) == DS41RT_STATUS_OK);
    check(ds41rt_copy_device_rows_async(dst, src, 17, 19, 16, 31, consumer) == DS41RT_STATUS_INVALID_ARGUMENT);
    check(ds41rt_copy_device_rows_async(dst, src, 17, 19, 43, 16, consumer) == DS41RT_STATUS_INVALID_ARGUMENT);
    check(ds41rt_copy_device_rows_async(dst, src, 17, 0, 43, 31, consumer) == DS41RT_STATUS_INVALID_ARGUMENT);
    check(ds41rt_copy_device_rows_async(dst, src, 0, 19, 43, 31, consumer) == DS41RT_STATUS_INVALID_ARGUMENT);
    check(ds41rt_copy_device_rows_async(dst, src, 17, std::numeric_limits<size_t>::max(), 43, 31, consumer) == DS41RT_STATUS_INVALID_ARGUMENT);
    check(ds41rt_copy_device_rows_async(dst, dst, 17, 19, 43, 31, consumer) == DS41RT_STATUS_INVALID_ARGUMENT);
    auto short_src = src;
    short_src.bytes = 18 * 31 + 16;
    check(ds41rt_copy_device_rows_async(dst, short_src, 17, 19, 43, 31, consumer) == DS41RT_STATUS_INVALID_ARGUMENT);
    for (auto input : {src, local}) {
      Gate gate;
      check(cudaLaunchHostFunc(static_cast<cudaStream_t>(pending), hold, &gate) == cudaSuccess);
      check(cudaMemsetAsync(dst.ptr, 255, bytes, static_cast<cudaStream_t>(consumer)) == cudaSuccess);
      check(ds41rt_copy_device_rows_async(dst, input, 17, 19, 43, 31, consumer) == DS41RT_STATUS_OK);
      check(ds41rt_cuda_stream_synchronize(consumer) == DS41RT_STATUS_OK);
      int32_t ready = 0;
      check(ds41rt_cuda_stream_query(pending, &ready) == DS41RT_STATUS_OK && ready == 0);
      gate.release.store(true, std::memory_order_release);
      check(ds41rt_cuda_stream_synchronize(pending) == DS41RT_STATUS_OK);
      check(ds41rt_copy_d2h(host.data(), dst, bytes) == DS41RT_STATUS_OK);
      for (size_t i = 0; i < bytes; ++i) {
        const size_t row = i / 43, column = i % 43;
        check(host[i] == (row < 19 && column < 17 ? values[row * 31 + column] : 255));
      }
    }
    check(ds41rt_free_device_buffer(&local) == DS41RT_STATUS_OK);
    check(ds41rt_cuda_stream_destroy(pending) == DS41RT_STATUS_OK);
    check(ds41rt_cuda_stream_destroy(consumer) == DS41RT_STATUS_OK);
    check(ds41rt_free_device_buffer(&dst) == DS41RT_STATUS_OK);
    check(ds41rt_cuda_set_device(source) == DS41RT_STATUS_OK);
    check(ds41rt_cuda_event_destroy(event) == DS41RT_STATUS_OK);
    check(ds41rt_cuda_stream_destroy(producer) == DS41RT_STATUS_OK);
    check(ds41rt_free_device_buffer(&src) == DS41RT_STATUS_OK);
  }
  std::cout << "bidirectional peer and local pitched copies preserve data/padding, reject invalid extents and do not join a pending lane\n";
}
