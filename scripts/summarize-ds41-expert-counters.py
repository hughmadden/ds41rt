#!/usr/bin/env python3
"""Extract fused expert metrics from Nsight Compute's raw wide CSV export.

System-memory fills count 32-byte L2 sectors. They are not pure weight reads
or a direct measurement of DRAM bandwidth / percentage of physical peak.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def summarize(path):
    with path.open(newline='') as source:
        reader = csv.DictReader(source)
        units = next(reader)
        scale = {'ns': 1e-9, 'us': 1e-6, 'ms': 1e-3, 's': 1.0}[units['gpu__time_duration.sum']]
        for key in ['lts__d_sectors_fill_sysmem.sum', 'lts__t_sectors.sum']:
            if units[key] != 'sector':
                raise ValueError(f'unexpected counter unit: {key}: {units[key]}')
        results = []
        for row in reader:
            if 'V41FusedSliceKernel' not in row['Kernel Name']:
                continue
            def value(key):
                number = float(row[key].replace(',', ''))
                if not math.isfinite(number) or number < 0:
                    raise ValueError(f'invalid metric: {key}')
                return number
            seconds = value('gpu__time_duration.sum') * scale
            if seconds <= 0:
                raise ValueError('kernel duration must be positive')
            fill = value('lts__d_sectors_fill_sysmem.sum') * 32
            traffic = value('lts__t_sectors.sum') * 32
            results.append(dict(kernel=row['Kernel Name'], grid=row['Grid Size'],
                                seconds=seconds, system_fill_bytes=fill,
                                system_fill_gb_per_second=fill/seconds/1e9,
                                l2_sector_access_bytes=traffic,
                                l2_sector_access_gb_per_second=traffic/seconds/1e9,
                                l2_hit_percent=value('lts__t_sector_hit_rate.pct'),
                                sm_throughput_percent=value('sm__throughput.avg.pct_of_peak_sustained_elapsed'),
                                registers_per_thread=value('launch__registers_per_thread')))
    if not results:
        raise ValueError('no fused V4.1 expert kernel records found')
    return dict(scope=__doc__, source=str(path),
                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), kernels=results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(summarize(args.csv), indent=2)+'\n')
