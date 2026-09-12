#!/usr/bin/env python3
"""Compare native image decoding/preparation with the pinned V4.1 reference.

Synthetic fixtures cover resampling, orientation, alpha, grayscale, palettes,
JPEG, WebP and GIF. This is component qualification, not a vision-serving pass.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
import PIL
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--reference-dir', type=Path, default=Path('/tmp/ds41-reference'))
parser.add_argument('--binary', type=Path, default=Path('rust/target/release/examples/v41-image-preprocess'))
parser.add_argument('--output-dir', type=Path, required=True)
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=False)
sys.path.insert(0, str(args.reference_dir / 'inference'))
import image_processor as reference

def digest(data): return hashlib.sha256(data).hexdigest()
ref_hash = digest((args.reference_dir / 'inference/image_processor.py').read_bytes())
assert ref_hash == '482759e3bcc4e9bb5ee582b244cc563f5d0e163d8b48dda91ebb7106e62f9272'
config = SimpleNamespace(vision_patch_size=14, vision_downsample_ratio=3,
    vision_min_pixels=544*544, vision_max_wh_ratio=None, vision_max_n_token=1024)
report = dict(scope=__doc__, reference_sha256=ref_hash, pillow=PIL.__version__,
    binary_sha256=digest(args.binary.read_bytes()), cases=[], passed=False)
def save(): (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2)+'\n')

def pattern(w, h):
    y,x,c = np.indices((h,w,3), dtype=np.uint32)
    return ((x*17+y*29+c*71+(x*y)%251)%256).astype(np.uint8)

process = subprocess.Popen([str(args.binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
try:
    fixtures = []
    for w,h in [(1,1),(43,91),(640,480),(1920,1080),(181,193),(544,544),(1,4096),(4096,1)]:
        fixtures.append((f'rgb-{w}x{h}',Image.fromarray(pattern(w,h)), 'PNG', {}))
    rgb = Image.fromarray(pattern(311,179))
    alpha = np.arange(311*179,dtype=np.uint32).reshape(179,311).astype(np.uint8)
    fixtures += [('rgba',Image.fromarray(np.dstack([np.asarray(rgb),alpha])), 'PNG', {}),
        ('gray',rgb.convert('L'),'PNG',{}),('gray16',Image.fromarray((np.arange(311*179,dtype=np.uint32).reshape(179,311)%65536).astype(np.uint16)),'PNG',{}),
        ('palette',rgb.quantize(colors=32),'PNG',{}),
        ('jpeg-444',rgb,'JPEG',dict(quality=90,subsampling=0)),
        ('jpeg-420',rgb,'JPEG',dict(quality=90,subsampling=2)),
        ('jpeg-progressive',rgb,'JPEG',dict(quality=90,progressive=True)),
        ('jpeg-gray',rgb.convert('L'),'JPEG',dict(quality=90)),
        ('jpeg-cmyk',rgb.convert('CMYK'),'JPEG',dict(quality=90)),
        ('webp-lossless',rgb,'WEBP',dict(lossless=True)),
        ('webp-lossy',rgb,'WEBP',dict(quality=90)),
        ('gif',rgb,'GIF',{})]
    # The reference converts to RGB without applying EXIF orientation.
    exif = Image.Exif(); exif[274] = 6
    fixtures.append(('exif-rotation',rgb,'JPEG',dict(quality=90,subsampling=0,exif=exif)))
    for name, image, fmt, options in fixtures:
        path = args.output_dir / f'{name}.{fmt.lower()}'
        image.save(path, format=fmt, **options)
        expected, vh,vw,lh,lw = reference.load_image(dict(data=path.read_bytes()),config)
        raw = expected.contiguous().view(torch.uint8).numpy().tobytes()
        actual_path = args.output_dir / f'{name}.bf16'
        start = time.monotonic()
        process.stdin.write(json.dumps(dict(path=str(path),patch_output=str(actual_path)))+'\n'); process.stdin.flush()
        actual = json.loads(process.stdout.readline())
        elapsed = time.monotonic()-start
        case = dict(name=name, format=fmt, input_sha256=digest(path.read_bytes()),
            reference_patch_sha256=digest(raw), native=actual, native_seconds=elapsed)
        report['cases'].append(case)
        if 'error' in actual:
            case['passed']=False
        else:
            data = actual_path.read_bytes()
            case['grid_equal'] = actual['grid'] == dict(pixel_height=vh*14,pixel_width=vw*14,
                vit_height=vh,vit_width=vw,llm_height=lh,llm_width=lw)
            case['patches_equal'] = raw == data
            a=np.frombuffer(data,dtype='<u2').astype(np.uint32)<<16
            b=np.frombuffer(raw,dtype='<u2').astype(np.uint32)<<16
            if a.shape == b.shape:
                diff = np.abs(a.view(np.float32)-b.view(np.float32))
                case['max_absolute_difference']=float(diff.max())
                case['mean_absolute_difference']=float(diff.mean())
                case['different_elements']=int(np.count_nonzero(diff))
            case['passed'] = case['grid_equal'] and case['patches_equal']
        save()
        print(name,case['passed'],case.get('max_absolute_difference'),flush=True)
    report['passed']=all(c['passed'] for c in report['cases']);save()
finally:
    process.stdin.close()
    assert process.wait(timeout=30)==0
raise SystemExit(0 if report['passed'] else 1)
