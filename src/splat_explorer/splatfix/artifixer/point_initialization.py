"""Stage COLMAP points for a fresh fit without importing source appearance."""
import math
from pathlib import Path
import shutil
import struct


def stage_points(source, destination, *, positions_only=False):
    source, destination = Path(source), Path(destination)
    if not positions_only:
        shutil.copyfile(source, destination)
        return {'positions': 'original_ply', 'colors': 'source_rgb',
                'gaussian_parameters': 'fresh; source scales, rotations and opacity excluded'}
    temporary = destination.with_suffix('.tmp')
    header = struct.Struct('<QdddBBBdQ')
    try:
        with source.open('rb') as reader, temporary.open('wb') as writer:
            raw = reader.read(8)
            if len(raw) != 8:
                raise ValueError('Truncated point-cloud header')
            count = struct.unpack('<Q', raw)[0]
            if count == 0:
                raise ValueError('Point-cloud initialization requires source positions')
            writer.write(raw)
            for _ in range(count):
                record = reader.read(header.size)
                if len(record) != header.size:
                    raise ValueError('Truncated point-cloud record')
                point = header.unpack(record)
                if not all(math.isfinite(value) for value in point[1:4]):
                    raise ValueError('Point-cloud positions must be finite')
                # COLMAP stores RGB as uint8. Neutral gray replaces all source
                # color; the trainer initializes Gaussian sizes from geometry.
                writer.write(record[:32] + bytes((128, 128, 128)) + record[35:])
                for _ in range(point[-1]):
                    track = reader.read(8)
                    if len(track) != 8:
                        raise ValueError('Truncated point-cloud track')
                    writer.write(track)
            if reader.read(1):
                raise ValueError('Unexpected data after point-cloud records')
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {'positions': 'original_ply', 'point_count': count,
            'colors': 'neutral_rgb_128; no source color or spherical harmonics',
            'gaussian_parameters': 'fresh; source scales, rotations and opacity excluded'}
