"""Verify GPU1 draft embedding reads GPU0's sole table with independent graphs.

Usage: python native/tests/v41_dspark_peer_embedding_selftest.py NATIVE_LIBRARY --width 7
Use --helpers LIBRARY when testing a standalone attention-ops build.
Requires CUDA PyTorch and two peer-accessible GPUs. No model files are needed.
"""
import argparse
import ctypes as C
import torch


def main():
    assert torch.cuda.device_count() >= 2
    parser = argparse.ArgumentParser()
    parser.add_argument("library")
    parser.add_argument("--width", type=int, choices=(5, 7), default=5)
    parser.add_argument("--helpers", help="Optional separate CUDA runtime helper library")
    args = parser.parse_args()
    width = args.width
    lib = C.CDLL(args.library)
    helpers = C.CDLL(args.helpers) if args.helpers else lib
    peer = helpers.ds41rt_cuda_enable_peer
    peer.argtypes, peer.restype = [C.c_int32], C.c_int32
    embed = lib.ds41rt_v41_dspark_embed_width
    embed.argtypes = [C.c_void_p] * 4 + [C.c_int32, C.c_int32, C.c_void_p]
    embed.restype = C.c_int32
    seed_ids = [0, 129279, 128799, 42, 500, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
    columns = torch.arange(5120, dtype=torch.int32)
    rows = {token: ((columns + token) % 251 - 125).to(torch.bfloat16) for token in seed_ids}
    # The full table exists only on GPU0. Populate every row that the cases read.
    with torch.cuda.device(0):
        table = torch.zeros((129280, 5120), device="cuda", dtype=torch.bfloat16)
        for token, row in rows.items():
            table[token].copy_(row)
        torch.cuda.synchronize()

    with torch.cuda.device(1):
        assert peer(0) == 0
        assert peer(0) == 0  # Planning can safely repeat peer admission.
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        ids = [torch.empty(16, device="cuda", dtype=torch.int32) for _ in streams]
        outputs = [torch.empty((16, width, 4, 5120), device="cuda", dtype=torch.bfloat16) for _ in streams]
        pres = [torch.empty((16, width, 4), device="cuda") for _ in streams]

        def launch(lane, count):
            assert embed(table.data_ptr(), ids[lane].data_ptr(), outputs[lane].data_ptr(),
                         pres[lane].data_ptr(), count, width, streams[lane].cuda_stream) == 0

        def check(lane, tokens):
            expected = torch.zeros((len(tokens), width, 4, 5120), dtype=torch.bfloat16)
            pre = torch.zeros((len(tokens), width, 4))
            for request, token in enumerate(tokens):
                if 0 <= token < 129280:
                    expected[request, 0] = rows[token]
                    expected[request, 1:] = rows[128799]
                    pre[request, :, 0] = 1
            assert torch.equal(outputs[lane][:len(tokens)].cpu(), expected)
            assert torch.equal(pres[lane][:len(tokens)].cpu(), pre)

        for count in [1, 3, 8, 16, 3]:
            tokens = [seed_ids[:count], list(reversed(seed_ids))[:count]]
            for lane in range(2):
                ids[lane][:count].copy_(torch.tensor(tokens[lane], dtype=torch.int32))
            torch.cuda.synchronize()
            for lane in range(2):
                launch(lane, count)
            for stream in streams:
                stream.synchronize()
            for lane in range(2):
                check(lane, tokens[lane])
            graphs = [torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()]
            for lane in range(2):
                with torch.cuda.graph(graphs[lane], stream=streams[lane]):
                    launch(lane, count)
            for iteration in range(3):
                changed = [list(reversed(value)) for value in tokens]
                if iteration == 2:
                    changed[0][0], changed[1][-1] = -1, 129280
                for lane in range(2):
                    ids[lane][:count].copy_(torch.tensor(changed[lane], dtype=torch.int32))
                torch.cuda.synchronize()
                allocated = torch.cuda.memory_allocated(1)
                for lane in range(2):
                    with torch.cuda.stream(streams[lane]):
                        graphs[lane].replay()
                for stream in streams:
                    stream.synchronize()
                assert torch.cuda.memory_allocated(1) == allocated
                for lane in range(2):
                    check(lane, changed[lane])
            print(f"PASS {count} requests: peer table, two independent graphs, changed IDs and invalid seeds", flush=True)


if __name__ == "__main__":
    main()
