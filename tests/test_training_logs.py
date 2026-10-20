import json

import pytest

from splat_explorer.web.training_logs import export_training_logs


def write_events(directory, records):
    pytest.importorskip('tensorboard')
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary
    from tensorboard.summary.writer.event_file_writer import EventFileWriter
    writer = EventFileWriter(str(directory))
    for step, wall_time, values in records:
        writer.add_event(Event(step=step, wall_time=wall_time, summary=Summary(value=[
            Summary.Value(tag=tag, simple_value=value) for tag, value in values.items()
        ])))
    writer.close()


def test_scalar_export_preserves_values_and_invalidates_cache(tmp_path):
    directory = tmp_path / 'artifixer3d/runs/splatfix/splatfix'
    write_events(directory, [(1, 100, {'loss/total/train': .5, 'num_particles/train': 20}),
                             (2, 104, {'loss/total/train': .25}),
                             (2, 105, {'loss/total/train': .125}),
                             (3, 106, {'loss/total/train': float('nan')})])
    logs, chart, error = export_training_logs(tmp_path)
    assert error is None
    data = json.loads(chart.read_text())
    assert len(data['points']) == 2
    assert data['points'][1]['loss/total/train'] == .125
    assert data['points'][1]['seconds'] == 4
    assert 'Iteration 2' in logs[0][1].read_text()
    assert 'loss/total/train=0.125' in logs[0][1].read_text()
    modified = chart.stat().st_mtime_ns
    export_training_logs(tmp_path)
    assert chart.stat().st_mtime_ns == modified
    write_events(directory, [(4, 110, {'loss/total/train': .0625})])
    logs, chart, error = export_training_logs(tmp_path)
    assert len(logs) == 2
    assert json.loads(chart.read_text())['points'][-1]['seconds'] == 10


def test_event_symlinks_cannot_escape_result(tmp_path):
    root = tmp_path / 'result'
    directory = root / 'artifixer3d/runs'
    directory.mkdir(parents=True)
    outside = tmp_path / 'events.out.tfevents.secret'
    outside.write_bytes(b'private')
    (directory / outside.name).symlink_to(outside)
    assert export_training_logs(root) == ([], None, None)
