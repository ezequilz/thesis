import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from splat_explorer.splatfix.checkpoint import atomic_json
from splat_explorer.splatfix.evaluation_chart import chart_data, write_evaluation_chart
from splat_explorer.splatfix import run_evaluation


def fixture(root):
    root.mkdir()
    poses = np.tile(np.eye(4), (5,1,1))
    poses[:,1,3] = [0,100,2,1,3]
    atomic_json(root/'result.json', {'output_dir': '/remote/run', 'inference_split': 'split.json'})
    atomic_json(root/'split.json', {'test': {'scene': {'transforms_path': 'poses.json',
        'selected_indices_path': 'refs.json', 'target_indices_path': 'targets.json'}}})
    atomic_json(root/'poses.json', {'frames': [{'transform_matrix': p.tolist()} for p in poses]})
    atomic_json(root/'refs.json', [1])
    atomic_json(root/'targets.json', [4,0,3,2])
    return root


def test_sorted_target_ids_exclude_context_and_scores_map_by_catalogue_index(tmp_path):
    root=fixture(tmp_path/'result')
    metrics={'per_image': [{'prepared_index': i, 'metrics': {'artifixer': {'psnr': v}}}
                           for i,v in [(4,17),(1,99),(0,20),(2,'Infinity')]]}
    data=chart_data(root,metrics)
    assert data['segments'][0]['indices']==[0,2,3,4]
    assert data['segments'][0]['heights']==[0,2,1,3]
    assert data['height_range']==3 and data['height_reversals']==2
    assert [r['index'] for r in data['scores']]==[0,2,4]
    write_evaluation_chart(root,metrics)
    assert Image.open(root/'trajectory-quality.png').size==(1500,1060)
    assert json.loads((root/'trajectory-quality.json').read_text())['target_count']==4


def test_no_ground_truth_omits_score_panel(tmp_path):
    root=fixture(tmp_path/'result')
    write_evaluation_chart(root)
    assert Image.open(root/'trajectory-quality.png').size==(1500,650)


def test_remote_paths_remapped_and_external_symlinks_rejected(tmp_path):
    root=fixture(tmp_path/'result')
    split=json.loads((root/'split.json').read_text())
    split['test']['scene']['transforms_path']='/remote/run/poses.json'
    atomic_json(root/'split.json',split)
    assert chart_data(root)['target_count']==4
    (root/'poses.json').rename(tmp_path/'outside.json')
    (root/'poses.json').symlink_to(tmp_path/'outside.json')
    with pytest.raises(ValueError,match='belong'):
        chart_data(root)


def test_saved_cv_segments_do_not_count_cuts_as_motion(tmp_path):
    root=fixture(tmp_path/'result')
    poses=json.loads((root/'poses.json').read_text())
    # CV camera Y is down; invert it to yield the same inferred world up.
    for f in poses['frames']:
        f['transform_matrix']=(np.asarray(f['transform_matrix']) @ np.diag([1,-1,-1,1])).tolist()
    atomic_json(root/'result.json',{})
    atomic_json(root/'request.json',{'trajectory':'trajectory.json',
                                   'inference_target_policy':'exclude_saved_anchor_indices'})
    atomic_json(root/'trajectory.json',{'camera_convention':'opencv_c2w','transforms':poses,
                'anchors':[{'frame_index':1}], 'segments':[{'start':0,'count':3},{'start':3,'count':2}]})
    data=chart_data(root)
    assert [s['indices'] for s in data['segments']]==[[0,2],[3,4]]
    assert data['height_reversals']==0 and data['vertical_travel']==4


def test_completed_benchmark_attempts_metrics_then_chart_without_hiding_failure(tmp_path,monkeypatch):
    root=fixture(tmp_path/'result')
    atomic_json(root/'benchmark-run.json',{})
    monkeypatch.setattr(run_evaluation,'runtime_environment',lambda cfg:{'PYTHONPATH':'compat'})
    calls=[]
    def worker(command,**kwargs):
        calls.append(command)
        raise RuntimeError('metric weights missing')
    monkeypatch.setattr(run_evaluation,'run_worker',worker)
    status=run_evaluation.finalize_run_evaluation(root,{'python':'python','repo':str(tmp_path)})
    assert calls[0][2]=='splat_explorer.splatfix.benchmark_evaluation'
    assert status['status']=='partial' and 'metric weights missing' in status['errors'][0]
    assert (root/'trajectory-quality.png').is_file()
    assert (root/'evaluation-status.json').is_file()


def test_cancel_prevents_postprocessing(tmp_path):
    with pytest.raises(InterruptedError):
        run_evaluation.finalize_run_evaluation(tmp_path,{},should_stop=lambda:True)


def test_job_worker_runs_evaluation_after_reconstruction(tmp_path, monkeypatch):
    from splat_explorer.splatfix import job_worker, benchmark
    atomic_json(tmp_path/'worker-request.json',{'stage':'benchmark','source':'source','output':'output',
                                               'runtime':{'python':'python','repo':'repo'}})
    monkeypatch.setattr(benchmark,'run_benchmark',lambda *a,**kw:{'output_dir':str(tmp_path/'result')})
    calls=[]
    monkeypatch.setattr(run_evaluation,'finalize_run_evaluation',lambda *a,**kw:calls.append(a))
    job_worker.execute(tmp_path)
    assert calls[0][0]==str(tmp_path/'result')
    assert json.loads((tmp_path/'worker-status.json').read_text())['status']=='completed'
