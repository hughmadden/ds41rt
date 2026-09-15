"""K5/K7 attention, layout and graph parity. Pass the native library path."""
import ctypes as C
import sys
import torch


def function(lib, name, pointers, integers):
    f = getattr(lib, name)
    f.argtypes = [C.c_void_p] * pointers + [C.c_int32] * integers + [C.c_void_p]
    f.restype = C.c_int32
    return f


def main():
    torch.manual_seed(701)
    torch.backends.cuda.matmul.allow_tf32 = False
    lib = C.CDLL(sys.argv[1])
    initialize = lib.ds41rt_v41_dspark_attention_initialize_width
    initialize.argtypes, initialize.restype = [C.c_int32], C.c_int32
    attend = function(lib, "ds41rt_v41_dspark_attention_fp8_width", 6, 3)
    legacy = function(lib, "ds41rt_v41_dspark_attention_fp8", 6, 2)
    layout = function(lib, "ds41rt_v41_dspark_terminal_layout_width", 4, 2)
    old_layout = function(lib, "ds41rt_v41_dspark_terminal_layout", 4, 1)
    assert initialize(6) != 0
    worst = 0.0
    for gpu in range(torch.cuda.device_count()):
        with torch.cuda.device(gpu):
            for width in (5, 7):
                assert initialize(width) == 0
                for requests in (1, 3, 8, 16):
                    stream = torch.cuda.Stream()
                    query = (torch.randn(requests, width, 64, 512, device="cuda") * .2).bfloat16()
                    # Exactly representable FP8 values and differing K32 scales.
                    packed = (torch.randint(-8, 9, (requests, 128, 512), device="cuda") / 8).to(torch.float8_e4m3fn)
                    scales = torch.randint(125, 129, (requests, 128, 16), device="cuda", dtype=torch.uint8)
                    ring = torch.cat((packed.view(torch.uint8), scales), dim=-1).contiguous()
                    values = packed.float() * torch.exp2(scales.float().repeat_interleave(32, -1) - 127)
                    draft = (torch.randn(requests, width, 512, device="cuda") * .2).bfloat16()
                    sink = torch.randn(64, device="cuda")
                    valid = [0, 1, 63, 64, 127, 128]
                    descriptors = torch.tensor([[requests - 1 - i, valid[i % len(valid)]] for i in range(requests)], device="cuda", dtype=torch.int32)
                    output = torch.empty_like(query)
                    args = [x.data_ptr() for x in (query, ring, draft, sink, descriptors, output)]
                    torch.cuda.synchronize()

                    def launch():
                        assert attend(*args, requests, requests, width, stream.cuda_stream) == 0

                    def check():
                        nonlocal worst
                        for i, (slot, count) in enumerate(descriptors.cpu().tolist()):
                            kv = torch.cat((values[slot, :count], draft[i].float()))
                            scores = query[i].float() @ kv.T / (512 ** .5)
                            probs = torch.softmax(torch.cat((scores, sink[None, :, None].expand(width, -1, -1)), -1), -1)[..., :-1]
                            expected = probs @ kv
                            error = (output[i].float() - expected).abs().max().item()
                            worst = max(worst, error)
                            # Native attention rounds probabilities to BF16 per
                            # 64-key chunk; this reference retains FP32 softmax.
                            torch.testing.assert_close(output[i].float(), expected, atol=.004, rtol=.025)

                    launch()
                    stream.synchronize()
                    check()
                    if width == 5:
                        old = torch.empty_like(output)
                        assert legacy(*args[:-1], old.data_ptr(), requests, requests, stream.cuda_stream) == 0
                        stream.synchronize()
                        assert torch.equal(old, output)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        launch()
                    for _ in range(2):
                        query.mul_(-1)
                        draft.mul_(-1)
                        torch.cuda.synchronize()
                        with torch.cuda.stream(stream):
                            graph.replay()
                        stream.synchronize()
                        check()
                    assert attend(*args, requests, requests, 6, stream.cuda_stream) != 0
                    assert attend(*args, 17, requests, width, stream.cuda_stream) != 0
                    assert attend(*args[:-1], query.data_ptr(), requests, requests, width, stream.cuda_stream) != 0

                    residual = torch.arange(requests * width * 20480, device="cuda", dtype=torch.int32).remainder(127).to(torch.bfloat16).reshape(requests, width, 4, 5120)
                    pre = torch.arange(requests * width * 4, device="cuda", dtype=torch.float32).reshape(requests, width, 4)
                    transposed = torch.empty((width, requests, 4, 5120), device="cuda", dtype=torch.bfloat16)
                    transposed_pre = torch.empty((width, requests, 4), device="cuda")
                    torch.cuda.synchronize()
                    pointers = [x.data_ptr() for x in (residual, pre, transposed, transposed_pre)]
                    assert layout(*pointers, requests, width, stream.cuda_stream) == 0
                    stream.synchronize()
                    assert torch.equal(transposed, residual.transpose(0, 1))
                    assert torch.equal(transposed_pre, pre.transpose(0, 1))
                    if width == 5:
                        transposed.fill_(float("nan"))
                        torch.cuda.synchronize()
                        assert old_layout(*pointers, requests, stream.cuda_stream) == 0
                        stream.synchronize()
                        assert torch.equal(transposed, residual.transpose(0, 1))
                    print(f"PASS GPU{gpu} K{width} R{requests}: reference, graph replay, layout, bounds", flush=True)
    print(f"Maximum attention absolute error: {worst:.6g}")


if __name__ == "__main__":
    main()
