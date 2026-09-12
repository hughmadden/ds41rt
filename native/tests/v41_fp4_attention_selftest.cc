#include "ds41rt_v41_sparse_attention.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

static void Check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
struct Buffer {
  void* allocation = nullptr;
  uint8_t* data = nullptr;
  size_t bytes;
  explicit Buffer(size_t size, bool unaligned = false) : bytes(size) {
    Check(cudaMalloc(&allocation, size + size_t(unaligned)));
    data = static_cast<uint8_t*>(allocation) + size_t(unaligned);
  }
  ~Buffer() { cudaFree(allocation); }
  Buffer(const Buffer&) = delete;
  void Fill(int value) { Check(cudaMemset(data, value, bytes)); }
  template<class T> void Copy(const std::vector<T>& values) {
    if (values.size()*sizeof(T) != bytes) throw std::runtime_error("test buffer size mismatch");
    Check(cudaMemcpy(data, values.data(), bytes, cudaMemcpyHostToDevice));
  }
};
static uint16_t BFloat16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return uint16_t((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}

int main() {
  Check(static_cast<cudaError_t>(ds41rt_v41_sparse_attention_initialize()));
  int checks = 0;
  for (int rows : {1, 6, 128, 256}) {
    Buffer query(size_t(rows)*64*512*2), output(query.bytes), sink(64*4);
    Buffer window(128*512, true), window_scales(128*16, true);
    Buffer proposal(size_t(rows)*512, true), proposal_scales(size_t(rows)*16, true);
    Buffer source(768*256, true), source_scales(768*32, true);
    Buffer private_source(3*256, true), private_scales(3*32, true);
    Buffer window_end(8), source_end(8), pages(8), metadata(size_t(rows)*10*8);
    Buffer selected(size_t(rows)*512*4), bounds(size_t(rows)*8);
    Buffer scratch(size_t(rows)*3*64*514*4);
    query.Fill(0); sink.Fill(0); window.Fill(0); proposal.Fill(0);
    window_scales.Fill(127); proposal_scales.Fill(127);
    source.Fill(0x22); private_source.Fill(0x22); // Four E2M1 ones per two bytes.
    source_scales.Fill(0x38); private_scales.Fill(0x38); // E4M3 scale one.
    window_end.Copy<uint64_t>({2048}); source_end.Copy<uint64_t>({512});
    pages.Copy<uint32_t>({1, 0});
    ds41rt_v41_sparse_kv_t view{};
    view.values[0]=window.data; view.values[1]=proposal.data;
    view.values[2]=source.data; view.values[3]=private_source.data;
    view.scales[0]=window_scales.data; view.scales[1]=proposal_scales.data;
    view.scales[2]=source_scales.data; view.scales[3]=private_scales.data;
    view.window_end=reinterpret_cast<const uint64_t*>(window_end.data);
    view.source_end=reinterpret_cast<const uint64_t*>(source_end.data);
    view.pages=reinterpret_cast<const uint32_t*>(pages.data);
    view.window_proposal_capacity=rows; view.source_capacity=768;
    view.source_proposal_capacity=3; view.page_stride=2; view.compressed=2;
    for (int parts : {0, 3}) for (int begin : {-1, 0, 2048, 2049}) for (int mode : {0, 1, 2}) {
      std::vector<uint64_t> meta;
      std::vector<int32_t> ids(size_t(rows)*512, -1);
      for (int row=0; row<rows; ++row) {
        const uint64_t fields[]={2048,0,uint64_t(rows),uint64_t(2048+row),0,
            uint64_t(mode?514:512),512,uint64_t(mode?2:0),uint64_t(mode==1?1:0),uint64_t(mode==2?2:1)};
        meta.insert(meta.end(), fields, fields+10);
        for (int key=0; key<(mode?2:512); ++key) ids[size_t(row)*512+key]=(mode?512:0)+key;
      }
      metadata.Copy(meta); selected.Copy(ids);
      bounds.Copy(std::vector<uint64_t>(rows, begin<0?0:uint64_t(begin)));
      output.Fill(0xcd);
      const auto* q=reinterpret_cast<const uint16_t*>(query.data);
      auto* out=reinterpret_cast<uint16_t*>(output.data);
      const auto* m=reinterpret_cast<const uint64_t*>(metadata.data);
      const auto* s=reinterpret_cast<const int32_t*>(selected.data);
      const auto* sk=reinterpret_cast<const float*>(sink.data);
      int status;
      if (begin>=0) status=ds41rt_v41_sparse_attention_bounded(q,sk,m,s,out,rows,0,&view,nullptr,
          reinterpret_cast<const uint64_t*>(bounds.data),parts?reinterpret_cast<float*>(scratch.data):nullptr,
          parts?scratch.bytes:0,parts?parts:1);
      else if (parts) status=ds41rt_v41_sparse_attention_split(q,sk,m,s,out,rows,0,&view,nullptr,
          reinterpret_cast<float*>(scratch.data),scratch.bytes,parts);
      else status=ds41rt_v41_sparse_attention(q,sk,m,s,out,rows,0,&view,nullptr);
      Check(static_cast<cudaError_t>(status));
      Check(cudaDeviceSynchronize());
      std::vector<uint16_t> actual(output.bytes/2);
      Check(cudaMemcpy(actual.data(), output.data, output.bytes, cudaMemcpyDeviceToHost));
      for (int row=0; row<rows; ++row) {
        // Zero queries/sink give equal attention weights. Window values are
        // zero, source values are one, and the sink contributes one to the denominator.
        int windows=begin==2048?std::min(row+1,128):128;
        int sources=mode?2:512;
        uint16_t expected=begin==2049?0:BFloat16(float(sources)/float(sources+windows+1));
        for (size_t col=0; col<64*512; ++col)
          if (actual[size_t(row)*64*512+col]!=expected)
            throw std::runtime_error("FP4 attention disagrees with closed-form reference");
      }
      ++checks;
    }
    std::cout << "PASS rows=" << rows << std::endl;
  }
  std::cout << "FP4 attention closed-form checks passed: " << checks << std::endl;
}
