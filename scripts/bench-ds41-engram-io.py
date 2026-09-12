#!/usr/bin/env python3
"""Compare completed CPU Engram row delivery; does not benchmark GPU or API latency.

Compile native/bench-engram-io.cpp with liburing, then pass --binary and --snapshot.
Cold trials request file-range eviction and report remaining resident pages; they
must not be described as fully cold unless resident_before is zero.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import random
import struct
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tokens', type=int, nargs='+', default=[1, 6, 18, 48])
    parser.add_argument('--modes', nargs='+', default=['mmap', 'willneed', 'populate', 'pread', 'uring', 'uring-fixed', 'uring-mmap', 'aio', 'posix-aio'])
    parser.add_argument('--cache', nargs='+', choices=['warm', 'fresh', 'cold', 'cold-global'], default=['warm', 'fresh', 'cold'],
                        help='cold requests file-range eviction; cold-global invokes installed system-wide cache-drop helper')
    parser.add_argument('--depth', type=int, default=64)
    parser.add_argument('--delivery', choices=['inline', 'gather', 'advisory'], default='inline')
    parser.add_argument('--lead-us', type=int, nargs='+', default=[0])
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--layer', type=int, choices=[1, 14], default=14)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    if args.delivery == 'inline' and args.lead_us != [0]:
        parser.error('lead time requires background delivery')
    if args.delivery == 'advisory' and any(mode not in ('willneed', 'uring-mmap') for mode in args.modes):
        parser.error('advisory delivery requires --modes willneed uring-mmap')
    prefix = f'layers.{args.layer}.engram.embed.'
    tensors = {}
    for path in sorted(args.snapshot.glob('*.safetensors')):
        with path.open('rb') as stream:
            size = struct.unpack('<Q', stream.read(8))[0]
            header = json.loads(stream.read(size))
        for key in ('weight', 'scale'):
            if prefix + key in header:
                tensor = header[prefix + key]
                tensors[key] = dict(path=str(path.resolve()), offset=8+size+tensor['data_offsets'][0], shape=tensor['shape'])
    if set(tensors) != {'weight', 'scale'} or tensors['weight']['path'] != tensors['scale']['path']:
        parser.error('expected weight and scale tensors in the same official checkpoint shard')
    weight, scale = tensors['weight'], tensors['scale']
    if weight['shape'][1:] != [256] or scale['shape'] != [weight['shape'][0], 8]:
        parser.error(f'unexpected Engram tensor shapes: {tensors}')
    trials = [(tokens, cache, repeat, mode, lead) for tokens in args.tokens for cache in args.cache
              for repeat in range(args.repeats) for mode in args.modes for lead in args.lead_us]
    random.Random(410).shuffle(trials)
    report = dict(scope='Synthetic uniform sorted unique Engram rows, one layer, CPU delivery only; excludes setup and duplicate scatter',
                  tensors=tensors, host=platform.node(), kernel=platform.release(),
                  binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(), records=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for tokens, cache, repeat, mode, lead in trials:
        command = [str(args.binary.resolve()), weight['path'], str(weight['offset']),
                   str(scale['offset']), str(weight['shape'][0]), str(tokens), mode,
                   cache, str(41000+repeat), str(args.depth)]
        if args.delivery != 'inline':
            command.extend([str(lead), args.delivery])
        result = subprocess.run(command, capture_output=True, text=True, timeout=180)
        if result.returncode:
            record = dict(tokens=tokens, cache=cache, mode=mode, repeat=repeat,
                          delivery=args.delivery, lead_us=lead, error=result.stderr.strip())
        else:
            record = json.loads(result.stdout)
            record['repeat'] = repeat
        report['records'].append(record)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(record), flush=True)
    if any('error' in record for record in report['records']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
