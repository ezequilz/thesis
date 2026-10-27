import struct

import pytest

from splat_explorer.splatfix.artifixer.point_initialization import stage_points


def test_geometry_only_keeps_exact_points_and_tracks_but_removes_source_colors(tmp_path):
    source, target = tmp_path / 'source.bin', tmp_path / 'target.bin'
    record = struct.Struct('<QdddBBBdQ')
    one = record.pack(17, 1.25, -2.5, .125, 20, 55, 220, .3, 1)
    track = struct.pack('<ii', 8, 3)
    two = record.pack(91, -8., 0., 33., 255, 1, 2, .7, 0)
    original = struct.pack('<Q', 2) + one + track + two
    source.write_bytes(original)
    provenance = stage_points(source, target, positions_only=True)
    expected = struct.pack('<Q', 2) + one[:32] + bytes([128]*3) + one[35:] + track + two[:32] + bytes([128]*3) + two[35:]
    assert target.read_bytes() == expected
    assert source.read_bytes() == original
    assert provenance['point_count'] == 2 and 'no source color' in provenance['colors']
    stage_points(source, target)
    assert target.read_bytes() == original


@pytest.mark.parametrize('data', [b'', struct.pack('<Q', 0), struct.pack('<Q', 1)])
def test_invalid_points_do_not_replace_previous_initialization(tmp_path, data):
    source, target = tmp_path / 'source.bin', tmp_path / 'target.bin'
    source.write_bytes(data)
    target.write_bytes(b'previous')
    with pytest.raises(ValueError):
        stage_points(source, target, positions_only=True)
    assert target.read_bytes() == b'previous'
