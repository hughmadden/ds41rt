#include "ds41rt_native.h"
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>
static void check(bool value) { if (!value) std::abort(); }
static void ok(ds41rt_status_t value) { check(value == DS41RT_STATUS_OK); }
int main() {
  constexpr size_t rows = 7, vocab = 129280;
  std::vector<float> logits(rows*vocab, -10.f);
  logits[17] = logits[91] = -2.f;
  logits[vocab*1+vocab-1] = 5.f;
  for (size_t row = 2; row <= 4; ++row) logits[row*vocab+91] = 4.f;
  logits[vocab*2+113] = std::numeric_limits<float>::quiet_NaN();
  logits[vocab*3+113] = std::numeric_limits<float>::infinity();
  logits[vocab*4+113] = -std::numeric_limits<float>::infinity();
  for (size_t i = 0; i < vocab; ++i) logits[vocab*5+i] = std::numeric_limits<float>::lowest();
  logits[vocab*6+255] = logits[vocab*6+256] = logits[vocab*6+65535] = 3.f;
  ds41rt_device_buffer_t input{}, indices{}, scores{};
  ok(ds41rt_alloc_device_buffer(logits.size()*4, &input));
  ok(ds41rt_alloc_device_buffer(rows*4, &indices));
  ok(ds41rt_alloc_device_buffer(rows*4, &scores));
  ok(ds41rt_copy_h2d(input, logits.data(), logits.size()*4));
  void* stream = nullptr; ok(ds41rt_cuda_stream_create(&stream));
  ok(ds41rt_cuda_logits_argmax_checked_f32_async(static_cast<const float*>(input.ptr),
     static_cast<uint32_t*>(indices.ptr), static_cast<float*>(scores.ptr), rows, vocab, stream));
  ok(ds41rt_cuda_stream_synchronize(stream));
  std::vector<uint32_t> actual(rows); std::vector<float> values(rows);
  ok(ds41rt_copy_d2h(actual.data(), indices, rows*4));
  ok(ds41rt_copy_d2h(values.data(), scores, rows*4));
  check(actual[0] == 17 && values[0] == -2.f);
  check(actual[1] == vocab-1 && values[1] == 5.f);
  for (size_t row = 2; row <= 4; ++row) check(std::isnan(values[row]));
  check(actual[5] == 0 && values[5] == std::numeric_limits<float>::lowest());
  check(actual[6] == 255 && values[6] == 3.f);
  ok(ds41rt_cuda_stream_destroy(stream));
  ok(ds41rt_free_device_buffer(&scores)); ok(ds41rt_free_device_buffer(&indices));
  ok(ds41rt_free_device_buffer(&input));
  std::cout << "checked argmax preserves ties and rejects every non-finite row\n";
}
