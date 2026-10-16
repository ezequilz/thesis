"""Saved method videos can be served and seeked without exposing other files."""
import io
from types import SimpleNamespace

import pytest

from splat_explorer.web.server import DashboardHandler
from test_splatfix_dashboard import studio


def test_saved_method_mp4_allowed_with_output_boundaries(studio, tmp_path):
    video = studio.root / 'run/gpu/results/oraclegs/seva/img2trajvid/scene/first-pass/samples-rgb.mp4'
    video.parent.mkdir(parents=True)
    video.write_bytes(b'video')
    assert studio.allowed_file(video) == video.resolve()
    outside = tmp_path / 'private.mp4'
    outside.write_bytes(b'private')
    link = video.parent / 'escape.mp4'
    link.symlink_to(outside)
    for path in (outside, link):
        with pytest.raises(ValueError):
            studio.allowed_file(path)
    private = video.with_suffix('.yaml')
    private.write_text('private config')
    with pytest.raises(ValueError, match='Unsupported artifact type'):
        studio.allowed_file(private)


@pytest.mark.parametrize('requested,status,body,content_range', [
    ('', 200, b'0123456789', None),
    ('bytes=0-1', 206, b'01', 'bytes 0-1/10'),
    ('bytes=4-', 206, b'456789', 'bytes 4-9/10'),
    ('bytes=-3', 206, b'789', 'bytes 7-9/10'),
    ('bytes=8-99', 206, b'89', 'bytes 8-9/10'),
    ('bytes=10-', 416, b'', 'bytes */10'),
    ('bytes=5-2', 416, b'', 'bytes */10'),
    ('bytes=-0', 416, b'', 'bytes */10'),
    ('bytes=nonsense', 416, b'', 'bytes */10'),
    ('bytes=0-1,4-5', 200, b'0123456789', None),
])
def test_video_stream_ranges(tmp_path, requested, status, body, content_range):
    video = tmp_path / 'samples-rgb.mp4'
    video.write_bytes(b'0123456789')
    headers, statuses = {}, []
    handler = SimpleNamespace(headers={'Range': requested}, wfile=io.BytesIO(),
                              send_response=statuses.append,
                              send_header=lambda k, v: headers.update({k: v}),
                              end_headers=lambda: None)
    DashboardHandler._send_video_file(handler, video)
    assert statuses == [status]
    assert handler.wfile.getvalue() == body
    assert headers['Content-Length'] == str(len(body))
    assert headers.get('Content-Range') == content_range
    if status != 416:
        assert headers['Content-Type'] == 'video/mp4'
        assert headers['Accept-Ranges'] == 'bytes'
