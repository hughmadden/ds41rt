#!/usr/bin/env python3
"""Create a test-only BF16 source reader from the pre-FP4 attention kernel.

The reference retains attention arithmetic and FP8 window reads, but directly
loads source rows independently decoded by Torch. It is not a serving kernel.
Use source from commit 4bc0e9a and compile with the same flags as the candidate.
"""
import argparse
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
source = a.source.read_text()
before = '          const uint8_t* source=v.values[tag]+physical*512+col;'
after = '''          if(tag>=2) {
            packed=*reinterpret_cast<const uint64_t*>(v.values[tag]+physical*1024+col*2);
          } else {
''' + before
assert source.count(before) == 1
source = source.replace(before, after)
before = '#endif\n        }\n        *reinterpret_cast<uint64_t*>(kv'
after = '#endif\n          }\n        }\n        *reinterpret_cast<uint64_t*>(kv'
assert source.count(before) == 1
source = source.replace(before, after)
a.output.write_text(source)
