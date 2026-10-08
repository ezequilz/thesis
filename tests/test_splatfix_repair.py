from pathlib import Path
import json
import struct
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from splat_explorer.rendering.base import Camera
from splat_explorer.splatfix.checkpoint import Checkpoint
from splat_explorer.splatfix import repair


def make_checkpoint(tmp_path):
    scene_path = tmp_path / 'scene.ply'
    scene_path.write_text('fixture')
    cp = Checkpoint.create(tmp_path / 'runs', scene_path, target_views=1,
                           metadata={'scene_load': {'min_opacity': .05, 'lod_level': 2},
                                     'renderer': {'backend': 'viser'}})
    camera = Camera.look_at(np.array([0., 0., -2.]), np.zeros(3), np.array([0., -1., 0.]), width=32, height=32)
    cp.add_view(np.full((32, 32, 3), 90, np.uint8), camera)
    return cp


class FakeRenderer:
    def __init__(self, scene):
        pass
    def render(self, camera):
        return (np.full((32, 32, 3), 90, np.uint8), np.ones((32, 32), np.float32), np.full((32, 32), 2., np.float32))


def mock_scene(monkeypatch):
    import splat_explorer.scene
    # These worker/orbit tests use fixed 32px fixtures. Exact-size preparation
    # and its run_repair wiring are exercised in test_splatfix_resolution.py.
    monkeypatch.setattr('splat_explorer.splatfix.resolution.prepare_repair_checkpoint',
                        lambda checkpoint, *args, **kwargs: checkpoint)
    calls = []
    monkeypatch.setattr(repair, 'capture_rgb', lambda *a, **kw: None)
    monkeypatch.setattr(repair, 'capture_plus_rgb', lambda *a, **kw: None)
    def load(path, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(means=np.zeros((2, 3)), colors=np.full((2, 3), .5))
    monkeypatch.setattr(splat_explorer.scene, 'load_scene', load)
    return calls


def test_cached_trajectory_is_identical_and_does_not_duplicate_anchor(tmp_path, monkeypatch):
    cp = make_checkpoint(tmp_path)
    calls = mock_scene(monkeypatch)
    root, first = repair.prepare_trajectory(cp, frames=9, renderer_factory=FakeRenderer)
    assert first['frames'][0]['rgb'] == first['anchors'][0]['original_rgb']
    assert first['frames'][0]['rgb'] != cp.views[0]['original_rgb']
    assert first['frames'][-1]['rgb'] == first['frames'][0]['rgb']
    assert len(list(root.glob('*.png'))) == 8
    assert calls == [{'min_opacity': .05, 'lod_level': 2}]
    again, second = repair.prepare_trajectory(cp, frames=9, renderer_factory=lambda _: pytest.fail('Cache must not render'))
    assert root == again and first == second
    (cp.root / first['frames'][1]['rgb']).write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='changed'):
        repair.prepare_trajectory(cp, frames=9)


def test_failed_render_can_be_retried(tmp_path, monkeypatch):
    cp = make_checkpoint(tmp_path)
    mock_scene(monkeypatch)
    with pytest.raises(InterruptedError):
        repair.prepare_trajectory(cp, frames=9, should_stop=lambda: True, renderer_factory=FakeRenderer)
    root, trajectory = repair.prepare_trajectory(cp, frames=9, renderer_factory=FakeRenderer)
    assert (root / 'trajectory.json').is_file()
    assert len(trajectory['frames']) == 9


def test_seed_is_point_cloud_not_original_gaussian_parameters(tmp_path):
    destination = tmp_path / 'points3D.bin'
    scene = SimpleNamespace(means=np.array([[1., 2., 3.]]), colors=np.array([[1., .5, 0.]]))
    repair.write_seed_points(destination, scene)
    data = destination.read_bytes()
    assert struct.unpack('<Q', data[:8]) == (1,)
    point = struct.unpack('<QdddBBBdQ', data[8:])
    assert point == (1, 1., 2., 3., 255, 127, 0, 0., 0)


def test_missing_edits_fail_before_gpu_or_api_work(tmp_path, monkeypatch):
    cp = make_checkpoint(tmp_path)
    monkeypatch.setattr(repair, 'validate_runtime', lambda _: pytest.fail('No GPU work before input validation'))
    with pytest.raises(ValueError, match='splatfix edit first'):
        repair.run_repair(cp.root, tmp_path / 'repairs')


@pytest.mark.parametrize('mode', ['baseline', 'edited'])
def test_replay_uses_selected_reference_and_all_three_official_phases(tmp_path, monkeypatch, mode):
    cp = make_checkpoint(tmp_path)
    mock_scene(monkeypatch)
    if mode == 'edited':
        edit = cp.root / 'views/000/repaired.png'
        Image.fromarray(np.full((32, 32, 3), 140, np.uint8)).save(edit)
        cp.views[0]['repaired_rgb'] = str(edit.relative_to(cp.root))
        cp.save()
    trajectory_root, trajectory = repair.prepare_trajectory(cp, frames=9, renderer_factory=FakeRenderer)
    monkeypatch.setattr(repair, 'validate_runtime', lambda runtime: repair.DEFAULT_RUNTIME)
    phases = []
    def worker(command, **kwargs):
        phase = command[-1]
        phases.append(phase)
        root = Path(command[command.index('--request') + 1]).parent
        req = json.loads((root / 'request.json').read_text())
        assert req['references'] == ([str(cp.image_path(cp.views[0], repaired=True))] if mode == 'edited'
                                     else [str(cp.root / trajectory['anchors'][0]['original_rgb'])])
        assert req['fit_iterations'] == 30000
        assert req['reconstruction_recipe']['custom_reconstruction_overrides'] is False
        if phase == 'caption':
            assert kwargs['env']['HF_HUB_OFFLINE'] == '0'
            caption = write_caption_fixture(root / 'caption.h5')
            (root / 'caption-result.json').write_text(json.dumps(caption))
        else:
            assert kwargs['env']['HF_HUB_OFFLINE'] == '1'
        if phase == 'plus':
            output = root / 'artifixer3d.ply'
            output.write_bytes(b'new isolated splat')
            (root / 'result.json').write_text(json.dumps({'splat_path': str(output), 'merged': False}))
    monkeypatch.setattr(repair, 'run_worker', worker)
    result = repair.run_repair(cp.root, tmp_path / 'repairs', mode=mode, frames=9, camera_scale=1., trajectory_mode='legacy_local_loops')
    assert phases == ['caption', 'infer', 'distill', 'plus']
    assert Path(result['splat_path']).read_bytes() == b'new isolated splat'
    assert Path(cp.manifest['scene_path']).read_text() == 'fixture'


@pytest.mark.parametrize('mode', ['baseline', 'edited'])
@pytest.mark.parametrize('loops', [1, 2])
def test_worker_calls_authors_fresh_reconstruction_and_exports_only_that_model(tmp_path, monkeypatch, mode, loops):
    """Exercise the actual adapter and its file/camera contract without CUDA."""
    import sys
    import types
    from splat_explorer.splatfix.official_worker import distill
    from splat_explorer.splatfix.checkpoint import camera_from_record

    cp = make_checkpoint(tmp_path)
    mock_scene(monkeypatch)
    _, trajectory = repair.prepare_trajectory(cp, frames=9, renderer_factory=FakeRenderer)
    if loops == 2:
        from copy import deepcopy
        extra = deepcopy(trajectory['transforms']['frames'])
        for frame in extra:
            frame['transform_matrix'][0][3] += 10.
        trajectory['transforms']['frames'].extend(extra)
        trajectory['frames'].extend(deepcopy(trajectory['frames']))
        trajectory['anchors'].append({**trajectory['anchors'][0], 'frame_index':9})
    root = tmp_path / mode
    root.mkdir()
    reference = cp.image_path(cp.views[0])
    if mode == 'edited':
        reference = tmp_path / 'edited.png'
        Image.fromarray(np.full((32, 32, 3), 125, np.uint8)).save(reference)
    request = {'mode': mode, 'seed': 42, 'references': [str(reference)] * loops,
               'source_points3d': str(cp.root / trajectory['points3d']), 'camera_scale': 1.}
    prediction_root = root / 'inference/splatfix/frames/batch_0000/pred'
    prediction_root.mkdir(parents=True)
    for index in range(9 * loops):
        Image.new('RGB', (32, 32), (index, 0, 0)).save(prediction_root / f'{index:05d}.png')
    events = []
    checkpoint = root / 'fresh.ckpt'
    output_render = root / 'renders'
    fresh_model = object()

    def module(name, **attrs):
        value = types.ModuleType(name)
        value.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, value)
        return value

    def train(scene, paths, **kwargs):
        events.append('train')
        assert kwargs == {
            'artifixer_frames_dir': root / 'distillation_predictions',
            'base_checkpoint': None, 'config_name': 'apps/colmap_3dgut_sparse_mcmc_lpips',
            'steps': 30000, 'use_wandb': False, 'replace': False}
        assert scene.reconstruction_checkpoint is None
        assert scene.has_gt is False
        assert scene.selected_indices == list(range(0, 8 * loops, 8))
        saved = json.loads(scene.transforms_path.read_text())
        assert saved['camera_model'] == 'OPENCV'
        assert len(saved['frames']) == 8 * loops  # Closing anchors are not extra targets.
        assert len({tuple(np.array(f['transform_matrix']).ravel()) for f in saved['frames']}) == 8 * loops
        mapping = json.loads((root / 'supervision.json').read_text())
        assert mapping['original_to_distillation'] == [8 * loop + i for loop in range(loops) for i in [0,1,2,3,4,5,6,7,0]]
        for index, original in enumerate(mapping['distillation_to_original']):
            target = kwargs['artifixer_frames_dir'] / f'{index:05d}.png'
            if index in scene.selected_indices:
                assert not target.exists()
            else:
                assert target.resolve() == prediction_root / f'{original:05d}.png'
        cv_c2w = camera_from_record(cp.views[0]).c2w
        gl_c2w = np.asarray(saved['frames'][0]['transform_matrix'])
        # Official COLMAP convention: flip @ inverse(OpenGL C2W).
        np.testing.assert_allclose(np.diag([1., -1., -1., 1.]) @ np.linalg.inv(gl_c2w),
                                   np.linalg.inv(cv_c2w), atol=1e-6)
        assert (scene.colmap_dir / 'images/anchor_00000.png').resolve() == reference.resolve()
        assert (scene.colmap_dir / 'sparse/0/points3D.bin').read_bytes() == Path(request['source_points3d']).read_bytes()
        checkpoint.write_bytes(b'fresh checkpoint')
        return checkpoint, False

    def render(scene, paths, **kwargs):
        events.append('render')
        assert kwargs['checkpoint'] == checkpoint
        assert kwargs['checkpoint_reused'] is False
        assert kwargs['render_trajectory_path'] == root / 'render_transforms_opengl.json'
        original_render = json.loads(kwargs['render_trajectory_path'].read_text())
        assert len(original_render['frames']) == len(trajectory['frames']) == 9 * loops
        assert original_render['frames'][0]['transform_matrix'] == original_render['frames'][8]['transform_matrix']
        for index, original in enumerate(trajectory['transforms']['frames']):
            np.testing.assert_allclose(original_render['frames'][index]['transform_matrix'],
                np.asarray(original['transform_matrix']) @ np.diag([1., -1., -1., 1.]))
        return output_render

    def pose(frame, applied):
        cv_w2c = np.diag([1., -1., -1., 1.]) @ np.linalg.inv(frame['transform_matrix'])
        return np.array([1., 0., 0., 0.]), cv_w2c[:3, 3]

    official = module('data_processing.artifixer3d', PreparedScene=SimpleNamespace,
        artifixer3d_paths=lambda scene, output_root, split, steps: SimpleNamespace(distillation_input_dir=root / 'distillation_input'),
        opencv_camera_from_mapping=lambda camera_id, mapping: mapping,
        colmap_pose_from_transforms_frame=pose,
        write_colmap_cameras=lambda path, cameras: path.write_text('calibrated cameras'),
        write_colmap_images=lambda path, images: path.write_text('calibrated images'),
        train_artifixer3d=train, render_artifixer3d=render)
    module('data_processing', artifixer3d=official)
    module('torch', manual_seed=lambda seed: None, cuda=SimpleNamespace(empty_cache=lambda: None))
    module('threedgrut')
    module('threedgrut.export')

    class Exporter:
        def export(self, model, output):
            events.append('export')
            assert model is fresh_model
            assert output == root / 'artifixer3d.ply'
            output.write_bytes(b'fresh model only')

    class Renderer:
        @staticmethod
        def from_checkpoint(**kwargs):
            assert kwargs['checkpoint_path'] == checkpoint
            return SimpleNamespace(model=fresh_model)

    module('threedgrut.export.ply_exporter', PLYExporter=Exporter)
    module('threedgrut.render', Renderer=Renderer)
    distill(root, request, trajectory)
    assert events == ['train', 'render', 'export']
    assert (root / 'artifixer3d.ply').read_bytes() == b'fresh model only'
    assert Path(cp.manifest['scene_path']).read_text() == 'fixture'
    assert json.loads((root / 'distillation.json').read_text())['splat_path'] == str(root / 'artifixer3d.ply')


def test_runtime_environment_only_adds_scoped_annotation_compatibility(monkeypatch):
    monkeypatch.setenv('PYTHONPATH', '/unexpected/global/packages')
    env = repair.runtime_environment(repair.DEFAULT_RUNTIME)
    compatibility = Path(env['PYTHONPATH'])
    assert compatibility == Path(repair.__file__).with_name('python_compat')
    assert (compatibility / 'sitecustomize.py').is_file()
    assert '/unexpected/global/packages' not in env['PYTHONPATH']
    assert env['HF_HUB_OFFLINE'] == '1'


def test_annotation_shim_preserves_native_self_and_backports_only_missing_symbol(monkeypatch, capsys):
    import runpy
    import typing
    import typing_extensions
    shim = str(Path(repair.__file__).with_name('python_compat') / 'sitecustomize.py')
    native = getattr(typing, 'Self', object())
    monkeypatch.setattr(typing, 'Self', native, raising=False)
    runpy.run_path(shim)
    assert typing.Self is native
    assert capsys.readouterr().err == ''
    monkeypatch.delattr(typing, 'Self')
    runpy.run_path(shim)
    assert typing.Self is typing_extensions.Self
    assert 'upstream source unchanged' in capsys.readouterr().err


@pytest.mark.parametrize('offline,expected', [(True, '1'), (False, '0')])
def test_runtime_environment_hub_mode_overrides_inherited_flags(monkeypatch, offline, expected):
    monkeypatch.setenv('HF_HUB_OFFLINE', 'inherited')
    monkeypatch.setenv('TRANSFORMERS_OFFLINE', 'inherited')
    env = repair.runtime_environment({**repair.DEFAULT_RUNTIME, 'model_hub_offline': offline})
    assert env['HF_HUB_OFFLINE'] == env['TRANSFORMERS_OFFLINE'] == expected


def test_runtime_environment_rejects_string_hub_mode():
    with pytest.raises(ValueError, match='model_hub_offline must be a boolean'):
        repair.runtime_environment({**repair.DEFAULT_RUNTIME, 'model_hub_offline': 'false'})


def test_native_library_directory_is_used_only_when_provisioned(tmp_path, monkeypatch):
    monkeypatch.setenv('LD_LIBRARY_PATH', '/container/cuda/lib')
    directory = tmp_path / 'native-libraries'
    cfg = {**repair.DEFAULT_RUNTIME, 'native_library_dir': str(directory)}
    assert repair.runtime_environment(cfg)['LD_LIBRARY_PATH'] == '/container/cuda/lib'
    directory.mkdir()
    assert repair.runtime_environment(cfg)['LD_LIBRARY_PATH'] == str(directory) + ':/container/cuda/lib'


def test_runtime_uses_pinned_compiler_and_revision_specific_extension_cache(tmp_path, monkeypatch):
    monkeypatch.setenv('PATH', '/legacy/slang/bin:/usr/bin')
    monkeypatch.setenv('TORCH_EXTENSIONS_DIR', '/workspace/python/torch_extensions')
    slang_bin = tmp_path / 'slang/bin'
    venv_bin = tmp_path / 'venv/bin'
    cfg = {**repair.DEFAULT_RUNTIME, 'python': str(venv_bin / 'python'),
           'slang_bin': str(slang_bin)}
    env = repair.runtime_environment(cfg)
    assert env['PATH'] == str(venv_bin) + ':/legacy/slang/bin:/usr/bin'
    assert env['TORCH_EXTENSIONS_DIR'] == '/workspace/artifixer-extensions/' + repair.UPSTREAM_REVISION[:12]
    slang_bin.mkdir(parents=True)
    env = repair.runtime_environment(cfg)
    assert env['PATH'] == str(slang_bin) + ':' + str(venv_bin) + ':/legacy/slang/bin:/usr/bin'
    assert env['TORCH_EXTENSIONS_DIR'] != '/workspace/python/torch_extensions'


def test_conditioning_matches_official_preparation_pose_contract_without_double_flip():
    from scipy.spatial.transform import Rotation
    from splat_explorer.splatfix.official_worker import opengl_transforms
    cv_w2cs = []
    for angles, translation in [([11, 24, -7], [1, 2, 3]), ([-2, 35, 4], [1.4, 2.3, 3.6])]:
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_euler('xyz', angles, degrees=True).as_matrix()
        pose[:3, 3] = translation
        cv_w2cs.append(pose)
    cv_c2ws = [np.linalg.inv(pose) for pose in cv_w2cs]
    trajectory = {'camera_convention': 'opencv_c2w', 'transforms': {
        'frames': [{'transform_matrix': pose.tolist()} for pose in cv_c2ws]}}
    converted = opengl_transforms(trajectory)
    # Exact official data_processing.camera_trajectories preparation formula:
    # opencv_w2c_to_opengl_c2w(world_to_camera) = inv(world_to_camera) @ flip.
    flip = np.diag([1., -1., -1., 1.])
    expected = [np.linalg.inv(pose) @ flip for pose in cv_w2cs]
    for frame, wanted in zip(converted['frames'], expected):
        np.testing.assert_allclose(frame['transform_matrix'], wanted)
    actual_relative = np.linalg.inv(converted['frames'][0]['transform_matrix']) @ converted['frames'][1]['transform_matrix']
    old_cv_relative = np.linalg.inv(cv_c2ws[0]) @ cv_c2ws[1]
    np.testing.assert_allclose(actual_relative, flip @ old_cv_relative @ flip)
    assert not np.allclose(actual_relative, old_cv_relative)
    assert converted == opengl_transforms({'transforms': converted})
    assert trajectory['transforms']['frames'][0]['transform_matrix'] == cv_c2ws[0].tolist()
    with pytest.raises(ValueError, match='Conflicting'):
        opengl_transforms({'camera_convention': 'opencv_c2w', 'transforms': converted})


@pytest.mark.parametrize('plus', [False, True])
@pytest.mark.parametrize('trajectory_mode', ['authors_orbit', 'legacy_local_loops'])
@pytest.mark.parametrize('reference_count', [1, 3, 6])
@pytest.mark.parametrize('insertion', [False, True])
@pytest.mark.parametrize('split_mode', ['single-split', 'double-split'])
def test_both_inference_passes_use_authors_opengl_conditioning(tmp_path, monkeypatch, plus, trajectory_mode, reference_count, split_mode, insertion):
    if insertion and trajectory_mode != 'authors_orbit':
        pytest.skip('Cache insertion coverage uses the dashboard authored orbit')
    import contextlib
    import sys
    import types
    from splat_explorer.splatfix.official_worker import inference
    cp = make_checkpoint(tmp_path)
    mock_scene(monkeypatch)
    _, trajectory = repair.prepare_trajectory(cp, frames=9, renderer_factory=FakeRenderer)
    trajectory['anchors'] = [{**trajectory['anchors'][0], 'frame_index': i} for i in range(reference_count)]
    root = tmp_path / 'repair'
    root.mkdir()
    if plus:
        rendered = root / 'rendered'
        for kind in ('renders', 'opacity'):
            (rendered / kind).mkdir(parents=True)
            for index in range(9):
                Image.fromarray(np.full((32, 32, 3), 128, np.uint8)).save(rendered / kind / f'{index:05d}.png')
        (root / 'distillation.json').write_text(json.dumps({'render_dir': str(rendered)}))
    request = {'runtime': {'checkpoint': 'weights.pt', 'model_id': 'wan'}, 'seed': 42,
               'checkpoint_root': str(cp.root), 'references': [str(cp.image_path(cp.views[0]))] * reference_count,
               'camera_scale': 1., 'mode': 'baseline', 'upstream_revision': 'pinned',
               'initialization': 'fixture', 'input_adaptation': 'fixture',
               'trajectory_mode': trajectory_mode}
    caption = write_caption_fixture(root / 'caption.h5')
    request['references'] = []
    for index in range(reference_count):
        path = root / f'edited-{index}.png'
        Image.new('RGB', (32, 32), (20 + index,) * 3).save(path)
        request['references'].append(str(path))
    request.update(caption_path=caption['caption_path'], caption_sha256=caption['sha256'])
    class Tensor(np.ndarray):
        def permute(self, *axes): return self.transpose(axes)
        def float(self): return self.astype(np.float32)
    def module(name, **attrs):
        value = types.ModuleType(name)
        value.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, value)
        return value
    module('torch', manual_seed=lambda _: None, device=lambda _: None, bool=np.bool_,
           from_numpy=lambda x: np.asarray(x).view(Tensor), stack=np.stack, tensor=np.asarray,
           ones=lambda count, dtype: np.ones(count, dtype=dtype), inference_mode=contextlib.nullcontext)
    transformer = SimpleNamespace(eval=lambda: SimpleNamespace(requires_grad_=lambda _: None))
    pipe = SimpleNamespace(transformer=transformer, clear_inference_caches=lambda: None,
                           vae=SimpleNamespace(config=SimpleNamespace(scale_factor_temporal=4)))
    seen = []
    target_indices = []
    context_indices = []
    items = []
    def conditioning(cameras, indices, neighbors, **kwargs):
        seen.append(cameras)
        target_indices.extend(indices)
        context_indices.extend(neighbors)
        return {}
    module('model_eval')
    parsed = []
    def get_pipe(args, device):
        # These settings must reach construction, not just the installed sampler:
        # upstream copies them into the transformer attention layers here.
        if insertion:
            assert parsed[parsed.index('--sink_size') + 1] == '1'
            assert parsed[parsed.index('--local_attn_size') + 1] == '21'
        else:
            assert '--sink_size' not in parsed and '--local_attn_size' not in parsed
        return pipe
    module('model_eval.run_inference', build_parser=lambda: SimpleNamespace(parse_args=lambda args: parsed.extend(args) or SimpleNamespace()),
           get_eval_pipe=get_pipe, process_item=lambda pipe, item, *args: items.append(item))
    module('model_eval.checkpoint_loading', load_transformer_checkpoint=lambda *args: None)
    module('model_training')
    module('model_training.data')
    loaded_prompts = []
    def load_prompt(paths):
        loaded_prompts.append(paths)
        return [None, 'fixture caption']
    module('model_training.data.utils', compute_camera_rays=conditioning, load_encoded_prompt=load_prompt)
    if trajectory_mode == 'authors_orbit':
        trajectory['segments'] = [{'indices': [0, 1, 2, 3, 4, 5, 6]}, {'indices': [7, 8]}]
        if split_mode == 'double-split':
            trajectory['segments'][0]['seed_index'] = 0
            trajectory['segments'][1].update(seed_index=reference_count - 1, indices=[8, 7])
    request['runtime']['image_cache_insertion'] = insertion
    installed = []
    monkeypatch.setattr('splat_explorer.splatfix.artifixer.official_worker.install_image_cache_insertion', lambda value: installed.append(value))
    if insertion and split_mode == 'single-split':
        trajectory['segments'][0]['cache_seed_index'] = 0
        trajectory['segments'][1].update(cache_seed_index=reference_count - 1, indices=[8, 7])
    resets = []
    pipe.clear_inference_caches = lambda: resets.append(len(items))
    inference(root, request, trajectory, plus=plus)
    assert installed == ([pipe] if insertion else [])
    assert parsed[parsed.index('--render_trajectory') + 1] == 'trajectory'
    assert loaded_prompts == [[Path(request['caption_path'])]]
    assert len(seen) == (2 if trajectory_mode == 'authors_orbit' else 1)
    assert resets == list(range(len(seen)))
    anchors = [anchor['frame_index'] for anchor in trajectory['anchors']]
    expected = [i for i in range(9) if trajectory_mode != 'authors_orbit' or i not in anchors]
    seeded = trajectory_mode == 'authors_orbit' and (split_mode == 'double-split' or insertion)
    if seeded:
        assert target_indices == [0, *[i for i in range(7) if i not in anchors], reference_count - 1, 8, 7]
        for item, seed in zip(items, [0, reference_count - 1]):
            assert item['frame_indices'][0] == seed
            assert not item['valid_frames_mask'][0]
            np.testing.assert_allclose(item['rgb_rendered'][0], (20 + seed) / 255)
            assert (item['opacity'][0] == 1).all()
    else:
        assert target_indices == expected
    assert context_indices == anchors * len(seen)
    assert sorted(i for item in items for i in item['frame_indices'][item['valid_frames_mask']].tolist()) == expected
    for item in items:
        assert len(item['rgb_rendered']) == len(item['opacity']) == len(item['frame_indices'])
        assert len(item['rgb_neighbors']) == len(anchors)
    cv = np.asarray(trajectory['transforms']['frames'][0]['transform_matrix'])
    np.testing.assert_allclose(seen[0]['frames'][0]['transform_matrix'], cv @ np.diag([1, -1, -1, 1]))
    assert seen[0]['camera_convention'] == 'opengl_c2w'


def test_scale_backprojection_uses_z_not_ray_distance_and_masks_invalid_geometry():
    from scipy.spatial.transform import Rotation
    from splat_explorer.splatfix.official_worker import sample_depth_points
    c2w = np.eye(4)
    c2w[:3, :3] = Rotation.from_euler('xyz', [12, 34, -20], degrees=True).as_matrix()
    c2w[:3, 3] = [4., 1., -2.]
    camera = {'w': 64, 'h': 32, 'fl_x': 30., 'fl_y': 31., 'cx': 32., 'cy': 16., 'transform_matrix': c2w.tolist()}
    depth = np.full((32, 64), 3.)
    opacity = np.ones_like(depth)
    depth[:4] = np.nan
    opacity[:, :4] = .2
    xy, world, pixels = sample_depth_points(depth, opacity, camera, max_samples=128)
    assert 32 <= len(xy) <= 128
    assert np.all(pixels[:, 0] >= 4) and np.all(pixels[:, 1] >= 4)
    recovered = (world - c2w[:3, 3]) @ c2w[:3, :3]
    np.testing.assert_allclose(recovered[:, 2], 3.)
    assert np.max(np.linalg.norm(recovered, axis=1)) > 3.5  # Off-axis ray distance differs.
    projected = recovered[:, :2] / recovered[:, 2, None] * [30, 31] + [32, 16]
    np.testing.assert_allclose(projected, xy)
    np.testing.assert_allclose(xy, pixels + .5)
    assert np.array_equal(world, sample_depth_points(depth, opacity, camera, max_samples=128)[1])
    with pytest.raises(ValueError, match='Insufficient'):
        sample_depth_points(depth, np.zeros_like(opacity), camera)


def scale_request(tmp_path, monkeypatch):
    cp = make_checkpoint(tmp_path)
    mock_scene(monkeypatch)
    trajectory_root, trajectory = repair.prepare_trajectory(cp, frames=9, renderer_factory=FakeRenderer)
    model = tmp_path / 'moge-model.pt'
    model.write_bytes(b'offline MoGe weights fixture')
    request = {'checkpoint_root': str(cp.root), 'trajectory': str(trajectory_root / 'trajectory.json'),
               'runtime': {'moge_model_path': str(model)}, 'upstream_revision': repair.UPSTREAM_REVISION,
               'references': ['/must-not-read-edited.png']}
    return cp, trajectory, request


def test_measurement_colmap_has_real_observations_and_roundtrips_depth(tmp_path, monkeypatch):
    from scipy.spatial.transform import Rotation
    from splat_explorer.splatfix.official_worker import measurement_colmap
    cp, trajectory, request = scale_request(tmp_path, monkeypatch)
    directory = tmp_path / 'measurement'
    counts = measurement_colmap(directory, request, trajectory)
    assert counts == [1024]
    assert (directory / 'images/anchor_00001.png').resolve() == cp.root / trajectory['anchors'][0]['original_rgb']
    with (directory / 'sparse/0/images.bin').open('rb') as stream:
        assert struct.unpack('<Q', stream.read(8))[0] == 1
        pose = struct.unpack('<idddddddi', stream.read(64))
        name = bytearray()
        while (char := stream.read(1)) != b'\0': name.extend(char)
        assert name == b'anchor_00001.png'
        count = struct.unpack('<Q', stream.read(8))[0]
        observations = [struct.unpack('<ddq', stream.read(24)) for _ in range(count)]
    with (directory / 'sparse/0/points3D.bin').open('rb') as stream:
        assert struct.unpack('<Q', stream.read(8))[0] == count
        points = [struct.unpack('<QdddBBBdQii', stream.read(59)) for _ in range(count)]
    rotation = Rotation.from_quat([*pose[2:5], pose[1]]).as_matrix()
    translation = np.array(pose[5:8])
    xyz = np.array([point[1:4] for point in points])
    recovered = xyz @ rotation.T + translation
    np.testing.assert_allclose(recovered[:, 2], 2., atol=1e-6)
    camera = trajectory['transforms']
    xy = recovered[:, :2] / recovered[:, 2, None] * [camera['fl_x'], camera['fl_y']] + [camera['cx'], camera['cy']]
    np.testing.assert_allclose(xy, np.array(observations)[:, :2], atol=1e-5)
    assert [p[0] for p in points] == [o[2] for o in observations]
    assert all(point[-3] == 1 and point[-2] == 1 and point[-1] == i for i, point in enumerate(points))
    assert Path(request['trajectory']).parent.joinpath('points3D.bin').read_bytes() != (directory / 'sparse/0/points3D.bin').read_bytes()


def mock_alignment(monkeypatch, *, metric=2.5, support=True, stats=None):
    import sys
    import types
    calls = []
    monkeypatch.delenv('MOGE_MODEL_PATH', raising=False)
    def align(**kwargs):
        calls.append(kwargs)
        assert kwargs['debug'] is False and kwargs['downsample_factor'] == 1
        assert (kwargs['colmap_dir'] / 'images.bin').is_file()
        return metric, stats or {'num_correspondences': 1024, 'rmse': .1}, [
            {'depths_metric': np.full(1024, 5.), 'mask': np.ones(1024) if support else np.zeros(1024)}]
    for name in ['data_processing', 'data_processing.sparse_recon', 'data_processing.sparse_recon.metric_alignment']:
        module = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules['data_processing.sparse_recon.metric_alignment'].align_colmap_to_metric_scale = align
    return calls


def test_scale_cache_is_shared_and_never_reads_edited_images(tmp_path, monkeypatch):
    from splat_explorer.splatfix.official_worker import measure_scale
    cp, trajectory, request = scale_request(tmp_path, monkeypatch)
    calls = mock_alignment(monkeypatch)
    first, second = tmp_path / 'baseline', tmp_path / 'edited'
    first.mkdir(); second.mkdir()
    measure_scale(first, request, trajectory)
    request.update(mode='edited', references=['/nonexistent/different/edit.png'])
    measure_scale(second, request, trajectory)
    assert len(calls) == 1
    before = json.loads((first / 'scale-result.json').read_text())
    after = json.loads((second / 'scale-result.json').read_text())
    assert before == after
    assert before['metric_scale'] == 2.5 and before['camera_scale'] == .025
    assert before['geometry_samples_per_anchor'] == [1024]
    assert set(before['recipe']['original_inputs_sha256']) == {
        trajectory['anchors'][0][key] for key in ('original_rgb', 'depth', 'opacity')}
    np.save(cp.root / trajectory['anchors'][0]['depth'], np.full((32, 32), 20.))
    with pytest.raises(ValueError, match='input changed'):
        measure_scale(second, request, trajectory)


def test_scale_binds_exact_hashed_model_and_restores_environment_on_failure(tmp_path, monkeypatch):
    import os
    import sys
    from splat_explorer.splatfix.official_worker import measure_scale
    _, trajectory, request = scale_request(tmp_path, monkeypatch)
    mock_alignment(monkeypatch)
    monkeypatch.setenv('MOGE_MODEL_PATH', '/unrelated/inherited-model.pt')
    def fail(**kwargs):
        assert os.environ['MOGE_MODEL_PATH'] == str(Path(request['runtime']['moge_model_path']).resolve())
        raise RuntimeError('alignment fixture failed')
    monkeypatch.setattr(sys.modules['data_processing.sparse_recon.metric_alignment'], 'align_colmap_to_metric_scale', fail)
    with pytest.raises(RuntimeError, match='alignment fixture failed'):
        measure_scale(tmp_path, request, trajectory)
    assert os.environ['MOGE_MODEL_PATH'] == '/unrelated/inherited-model.pt'


def test_scale_rejects_empty_anchors_before_calling_official_alignment(tmp_path, monkeypatch):
    from splat_explorer.splatfix.official_worker import measure_scale
    _, trajectory, request = scale_request(tmp_path, monkeypatch)
    calls = mock_alignment(monkeypatch)
    trajectory['anchors'] = []
    with pytest.raises(ValueError, match='nonempty original anchor observations'):
        measure_scale(tmp_path, request, trajectory)
    assert calls == []


@pytest.mark.parametrize('metric,support', [(float('nan'), True), (0., True), (2.5, False)])
def test_invalid_automatic_scale_never_falls_back_to_one(tmp_path, monkeypatch, metric, support):
    from splat_explorer.splatfix.official_worker import measure_scale
    cp, trajectory, request = scale_request(tmp_path, monkeypatch)
    mock_alignment(monkeypatch, metric=metric, support=support)
    root = tmp_path / 'result'
    root.mkdir()
    with pytest.raises(ValueError, match='invalid|Insufficient'):
        measure_scale(root, request, trajectory)
    assert not (root / 'scale-result.json').exists()


def test_default_repair_measures_before_inference_and_reuses_identical_scale(tmp_path, monkeypatch):
    cp, trajectory, request = scale_request(tmp_path, monkeypatch)
    monkeypatch.setattr(repair, 'validate_runtime', lambda runtime: repair.DEFAULT_RUNTIME)
    phases = []
    def worker(command, **kwargs):
        phase = command[-1]
        phases.append(phase)
        path = Path(command[command.index('--request') + 1])
        body = json.loads(path.read_text())
        if phase == 'scale':
            assert body['camera_scale'] is None
            (path.parent / 'scale-result.json').write_text(json.dumps({'metric_scale': 2.5}))
        else:
            assert body['camera_scale'] == .025
            assert 'official MoGe' in body['camera_scale_provenance']
        if phase == 'caption':
            caption = write_caption_fixture(path.parent / 'caption.h5')
            (path.parent / 'caption-result.json').write_text(json.dumps(caption))
        if phase == 'plus':
            splat = path.parent / 'artifixer3d.ply'
            splat.write_bytes(b'fresh output')
            (path.parent / 'result.json').write_text(json.dumps({'splat_path': str(splat)}))
    monkeypatch.setattr(repair, 'run_worker', worker)
    repair.run_repair(cp.root, tmp_path / 'runs', mode='baseline', frames=9, trajectory_mode='legacy_local_loops')
    assert phases == ['scale', 'caption', 'infer', 'distill', 'plus']


def test_official_finite_scale_survives_nonfinite_ancillary_diagnostics(tmp_path, monkeypatch):
    from splat_explorer.splatfix.official_worker import measure_scale
    _, trajectory, request = scale_request(tmp_path, monkeypatch)
    mock_alignment(monkeypatch, stats={'mean_error': float('inf'), 'weighted_mean_error': float('nan'), 'median_error': .1})
    root = tmp_path / 'result'
    root.mkdir()
    measure_scale(root, request, trajectory)
    result = json.loads((root / 'scale-result.json').read_text())
    assert result['metric_scale'] == 2.5
    assert result['stats'] == {'mean_error': None, 'weighted_mean_error': None, 'median_error': .1}
    assert set(result['nonfinite_statistics']) == {'mean_error', 'weighted_mean_error'}


def test_single_gpu_worker_does_not_inherit_partial_distributed_launch(monkeypatch):
    monkeypatch.setenv('RANK', '0')
    monkeypatch.setenv('LOCAL_RANK', '0')
    monkeypatch.delenv('WORLD_SIZE', raising=False)
    monkeypatch.setenv('MASTER_ADDR', 'unrelated-container-master')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '3')
    env = repair.runtime_environment(repair.DEFAULT_RUNTIME)
    assert not {'RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR'} & env.keys()
    assert env['CUDA_VISIBLE_DEVICES'] == '3'
    assert __import__('os').environ['RANK'] == '0'


def write_caption_fixture(path, *, count=1):
    h5py = pytest.importorskip('h5py')
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, 'w') as output:
        dataset = output.create_dataset('anchor_00000.png', data=np.full((4, 4096), 16256, np.uint16))
        dataset.attrs['caption'] = 'A scene caption from original anchors.'
        dataset.attrs['image_indices'] = np.arange(count)
    return {'caption_path': str(path), 'sha256': repair.digest_file(path)}


@pytest.mark.parametrize('anchor_count', [1, 6])
def test_caption_cache_originals_only_exact_models_and_both_modes(tmp_path, monkeypatch, anchor_count):
    import sys
    import types
    from splat_explorer.splatfix import official_worker as worker
    cp, trajectory, request = scale_request(tmp_path, monkeypatch)
    request['runtime']['model_id'] = 'wan-test'
    trajectory['anchors'] = [{**trajectory['anchors'][0], 'view_id': str(i)} for i in range(anchor_count)]
    model_calls = []
    def snapshot(model_id, **kwargs):
        model_calls.append((model_id, kwargs))
        return Path('/frozen') / model_id, {'model_id': model_id, 'revision': 'immutable', 'files': []}
    monkeypatch.setattr(worker, 'caption_model_snapshot', snapshot)
    generated = []
    def generate(**kwargs):
        generated.append(kwargs)
        transforms = json.loads((kwargs['input_path'] / 'transforms.json').read_text())
        frames = transforms['frames']
        assert len(frames) == anchor_count
        assert (kwargs['input_path'] / frames[0]['file_path']).resolve() == cp.root / trajectory['anchors'][0]['original_rgb']
        assert kwargs['captioning_model_id'] == '/frozen/' + worker.CAPTION_MODEL_ID
        assert kwargs['text_encoder_model_id'] == '/frozen/wan-test'
        assert kwargs['dataset_downsample_factor'] == 1
        write_caption_fixture(kwargs['output_path'], count=anchor_count)
    for name in ['data_processing', 'data_processing.captioning', 'data_processing.captioning.generate_captions']:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules['data_processing.captioning.generate_captions'].generate_caption_hdf5 = generate
    first, second = tmp_path / 'baseline-caption', tmp_path / 'edited-caption'
    first.mkdir(); second.mkdir()
    worker.prepare_caption(first, request, trajectory)
    request.update(mode='edited', references=['/nonexistent/edit.png'])
    worker.prepare_caption(second, request, trajectory)
    assert len(generated) == 1
    before = json.loads((first / 'caption-result.json').read_text())
    assert before == json.loads((second / 'caption-result.json').read_text())
    assert before['recipe']['originals'][0]['rgb'] == trajectory['anchors'][0]['original_rgb']
    Path(before['caption_path']).write_bytes(b'corrupt cache')
    with pytest.raises(ValueError, match='checksum mismatch'):
        worker.prepare_caption(second, request, trajectory)
    assert len(generated) == 1


def test_caption_validation_rejects_nonfinite_empty_text_and_wrong_shape(tmp_path):
    h5py = pytest.importorskip('h5py')
    from splat_explorer.splatfix.official_worker import validate_caption
    path = tmp_path / 'caption.h5'
    for data, caption in [(np.full((3,4096), 32640, np.uint16), 'nonfinite'),
                          (np.zeros((3,4096), np.uint16), ''),
                          (np.zeros((3,40), np.uint16), 'wrong width')]:
        with h5py.File(path, 'w') as output:
            output.create_dataset('anchor', data=data).attrs['caption'] = caption
        with pytest.raises(ValueError, match='Invalid authors caption'):
            validate_caption(path)


def test_caption_snapshot_resolves_locally_and_pins_only_wan_text_components(tmp_path, monkeypatch):
    import sys
    import types
    from splat_explorer.splatfix.official_worker import caption_model_snapshot
    snapshot = tmp_path / 'snapshots' / 'commit'
    for relative in ['text_encoder/model.safetensors', 'tokenizer/tokenizer.json', 'transformer/model.safetensors']:
        file = snapshot / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(relative.encode())
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        return str(snapshot)
    module = types.ModuleType('huggingface_hub')
    module.snapshot_download = download
    monkeypatch.setitem(sys.modules, 'huggingface_hub', module)
    path, identity = caption_model_snapshot('wan', text_only=True)
    assert path == snapshot
    assert calls == [{'repo_id': 'wan', 'local_files_only': True,
                      'allow_patterns': ['tokenizer/*', 'text_encoder/*']}]
    assert [f['path'] for f in identity['files']] == ['text_encoder/model.safetensors', 'tokenizer/tokenizer.json']
    assert identity['revision'] == 'commit'
    (snapshot / 'model-00001-of-00001.safetensors').write_bytes(b'qwen fixture')
    caption_model_snapshot('qwen')
    patterns = calls[-1]['allow_patterns']
    assert 'model*.safetensors*' in patterns and 'tokenizer*' in patterns
    assert 'preprocessor_config.json' in patterns and 'video_preprocessor_config.json' in patterns
    assert '*' not in patterns and 'README.md' not in patterns and 'assets/*' not in patterns


def test_failed_caption_preparation_can_retry_without_partial_cache(tmp_path, monkeypatch):
    import sys
    import types
    from splat_explorer.splatfix import official_worker as worker
    cp, trajectory, request = scale_request(tmp_path, monkeypatch)
    request['runtime']['model_id'] = 'wan-test'
    monkeypatch.setattr(worker, 'caption_model_snapshot',
                        lambda *args, **kwargs: (Path('/frozen'), {'files': []}))
    calls = []
    def generate(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            kwargs['output_path'].write_bytes(b'incomplete')
            raise RuntimeError('fixture interrupted generation')
        write_caption_fixture(kwargs['output_path'])
    for name in ['data_processing', 'data_processing.captioning', 'data_processing.captioning.generate_captions']:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules['data_processing.captioning.generate_captions'].generate_caption_hdf5 = generate
    with pytest.raises(RuntimeError, match='fixture interrupted'):
        worker.prepare_caption(tmp_path, request, trajectory)
    assert not (tmp_path / 'caption-result.json').exists()
    assert not list((cp.root / 'captions').glob('.*'))
    worker.prepare_caption(tmp_path, request, trajectory)
    assert len(calls) == 2
    assert (tmp_path / 'caption-result.json').is_file()


@pytest.mark.parametrize('layout', ['standard', 'sharded'])
def test_caption_snapshot_recognizes_only_in_cache_blob_addresses(tmp_path, monkeypatch, layout):
    import sys
    import types
    from splat_explorer.splatfix.official_worker import caption_model_snapshot
    cache = tmp_path / 'hub'
    repo = cache / 'models--Qwen--fixture'
    snapshot = repo / 'snapshots' / ('a' * 40)
    snapshot.mkdir(parents=True)
    address = '69' + 'b' * 62
    blob = repo / 'blobs' / address if layout == 'standard' else cache / 'blobs' / address[:2] / address
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b'weights fixture')
    if layout == 'sharded':
        alias = repo / 'blobs' / ('c' * 64)
        alias.parent.mkdir(parents=True)
        alias.symlink_to(blob)
    else:
        alias = blob
    (snapshot / 'model-00001-of-00001.safetensors').symlink_to(alias)
    outside = tmp_path / 'unrelated/blobs' / ('d' * 64)
    outside.parent.mkdir(parents=True)
    outside.write_bytes(b'external weights fixture')
    (snapshot / 'model-00002-of-00002.safetensors').symlink_to(outside)
    module = types.ModuleType('huggingface_hub')
    module.snapshot_download = lambda **kwargs: str(snapshot)
    monkeypatch.setitem(sys.modules, 'huggingface_hub', module)
    _, identity = caption_model_snapshot('Qwen/fixture')
    inside, external = identity['files']
    assert inside['content_address'] == address and inside['copied_file_mtime_ns'] is None
    assert external['content_address'] is None and external['copied_file_mtime_ns'] is not None


def test_blob_address_rejects_wrong_shard_other_repository_and_escaped_cache(tmp_path):
    from splat_explorer.splatfix.official_worker import hub_blob_address
    cache = tmp_path / 'hub'
    snapshot = cache / 'models--Qwen--fixture/snapshots' / ('a' * 40)
    snapshot.mkdir(parents=True)
    address = '69' + 'b' * 62
    assert hub_blob_address(snapshot, cache / 'blobs/ff' / address, 'Qwen/fixture') is None
    assert hub_blob_address(snapshot, cache / 'models--Other/blobs' / address, 'Qwen/fixture') is None
    outside = tmp_path / 'outside'
    outside.mkdir()
    (cache / 'blobs').symlink_to(outside, target_is_directory=True)
    assert hub_blob_address(snapshot, cache / 'blobs/69' / address, 'Qwen/fixture') is None


def test_unique_supervision_resolves_cross_loop_overlap_without_dropping_nearby_views(tmp_path):
    from copy import deepcopy
    from splat_explorer.splatfix.official_worker import unique_supervision
    a, b, nearby = np.eye(4), np.eye(4), np.eye(4)
    b[0, 3] = 2.
    nearby[0, 3] = np.nextafter(0., 1.)  # Deliberately below any practical tolerance.
    trajectory = {'camera_convention': 'opencv_c2w', 'transforms': {
        'w':32,'h':32,'fl_x':20.,'fl_y':20.,'cx':16.,'cy':16.,
        'frames': [{'transform_matrix': m.tolist()} for m in (a,b,a,nearby,b)]},
        'anchors': [{'frame_index':2}, {'frame_index':1}, {'frame_index':4}]}
    references = [tmp_path / 'a.png', tmp_path / 'b.png', tmp_path / 'same-b.png']
    for path, color in zip(references, ('red','blue','blue')):
        Image.new('RGB',(32,32),color).save(path)
    # Encoding differences alone do not create conflicting RGB supervision.
    Image.new('RGB',(32,32),'blue').save(references[-1], compress_level=0)
    original = deepcopy(trajectory)
    transforms, mapping = unique_supervision(trajectory, references)
    assert mapping['original_to_distillation'] == [0,1,0,2,1]
    assert mapping['distillation_to_original'] == [2,4,3]
    assert mapping['selected_indices'] == [0,1]
    assert [g['reference'] is not None for g in mapping['groups']] == [True,True,False]
    assert len(transforms['frames']) == 3
    assert trajectory == original  # Plus and diffusion retain original indices.
    Image.new('RGB',(32,32),'green').save(references[-1])
    with pytest.raises(ValueError, match='Conflicting saved reference RGB'):
        unique_supervision(trajectory, references)


def test_author_orbit_requires_multiple_distinct_anchors_before_runtime(tmp_path, monkeypatch):
    cp = make_checkpoint(tmp_path)
    monkeypatch.setattr(repair, 'validate_runtime', lambda _: pytest.fail('Single anchor must fail before runtime'))
    with pytest.raises(ValueError, match='at least two distinct'):
        repair.run_repair(cp.root, tmp_path / 'output', mode='baseline')


@pytest.mark.parametrize('mode', ['baseline', 'edited'])
def test_author_orbit_scale_precedes_one_shared_temporal_path(tmp_path, monkeypatch, mode):
    from dataclasses import replace
    from splat_explorer.splatfix.checkpoint import camera_from_record
    cp = make_checkpoint(tmp_path)
    cp.manifest['target_views'] = 2
    cp.save()
    camera = camera_from_record(cp.views[0])
    cp.add_view(np.full((32, 32, 3), 91, np.uint8), replace(camera, position=camera.position + [1, 0, 0]))
    mock_scene(monkeypatch)
    from splat_explorer.splatfix import rendering
    monkeypatch.setattr(rendering, 'BundleRenderer', FakeRenderer)
    monkeypatch.setattr(repair, 'validate_runtime', lambda _: repair.DEFAULT_RUNTIME)
    if mode == 'edited':
        for view in cp.views:
            path = cp.image_path(view).with_name('repaired.png')
            Image.new('RGB', (32, 32), (140, 140, 140)).save(path)
            view['repaired_rgb'] = str(path.relative_to(cp.root))
        cp.save()
    events, roots = [], []
    def worker(command, **kwargs):
        if '--full-output' in command:
            events.append('orbit')
            assert command[command.index('--split-mode') + 1] == 'double-split'
            assert float(command[command.index('--metric-scale') + 1]) == 2.5
            source = json.loads(Path(command[command.index('--transforms') + 1]).read_text())
            poses = np.array([frame['transform_matrix'] for frame in source['frames']])
            middle = poses[0].copy()
            middle[:3, 3] = (poses[0, :3, 3] + poses[1, :3, 3]) / 2
            full = [pose @ np.diag([1., -1., -1., 1.]) for pose in [poses[0], middle, poses[1]]]
            Path(command[command.index('--full-output') + 1]).write_text(json.dumps({'frames': [{'transform_matrix': pose.tolist()} for pose in full]}))
            Path(command[command.index('--provenance') + 1]).write_text(json.dumps({'implementation': 'authors fixture', 'source_transforms': str(source), 'segments': [{'target_indices': [0], 'full_frame_indices': [1], 'seed_full_frame_index': 0}]}))
            return
        phase = command[-1]
        events.append(phase)
        path = Path(command[command.index('--request') + 1])
        req = json.loads(path.read_text())
        trajectory = json.loads(Path(req['trajectory']).read_text())
        if mode == 'baseline':
            assert req['references'] == [str(cp.root / a['original_rgb']) for a in trajectory['anchors']]
            assert req['reference_sha256'] == [repair.digest_file(p) for p in req['references']]
        if phase == 'scale':
            assert len(trajectory['frames']) == 2
            assert [anchor['frame_index'] for anchor in trajectory['anchors']] == [0, 1]
            (path.parent / 'scale-result.json').write_text(json.dumps({'metric_scale': 2.5}))
            return
        assert len(trajectory['frames']) == 3
        assert trajectory['segments'] == [{'indices': [1], 'seed_index': 0}]
        assert [anchor['frame_index'] for anchor in trajectory['anchors']] == [0, 2]
        assert req['legacy_trajectory_parameters']['used'] is False
        assert req['camera_scale'] == .025
        roots.append(req['trajectory'])
        if phase == 'caption':
            (path.parent / 'caption-result.json').write_text(json.dumps(write_caption_fixture(path.parent / 'caption.h5')))
        if phase == 'plus':
            splat = path.parent / 'artifixer3d.ply'
            splat.write_bytes(b'fresh output')
            (path.parent / 'result.json').write_text(json.dumps({'splat_path': str(splat)}))
    monkeypatch.setattr(repair, 'run_worker', worker)
    repair.run_repair(cp.root, tmp_path / 'output', mode=mode)
    assert events == ['scale', 'orbit', 'caption', 'infer', 'distill', 'plus']
    assert len(set(roots)) == 1
    # This shared cache contains original renders even when the edited arm ran.
    trajectory = json.loads(Path(roots[0]).read_text())
    assert [trajectory['frames'][i]['rgb'] for i in [0, 2]] == [a['original_rgb'] for a in trajectory['anchors']]
    assert all(a['original_rgb'] != v['original_rgb'] for a, v in zip(trajectory['anchors'], cp.views))
    same_root, _ = repair.prepare_saved_path(cp, orbit=trajectory['recipe']['orbit'], renderer_factory=lambda _: pytest.fail('Cache must be shared'))
    assert same_root == Path(roots[0]).parent


@pytest.mark.parametrize('kind', ['legacy', 'saved'])
def test_repair_captures_viser_rgb_instead_of_selection_or_cuda_rgb(tmp_path, monkeypatch, kind):
    cp = make_checkpoint(tmp_path)
    mock_scene(monkeypatch)
    preview = cp.image_path(cp.views[0])
    before = preview.read_bytes()
    class SourceRenderer(FakeRenderer):
        def render(self, camera):
            rgb, alpha, depth = super().render(camera)
            return np.full_like(rgb, 210), alpha, depth
    def capture(scene, cameras, destinations, **kwargs):
        for camera, path in zip(cameras, destinations):
            Image.new('RGB', (camera.width, camera.height), (217, 217, 217)).save(path)
    monkeypatch.setattr(repair, 'capture_rgb', capture)
    if kind == 'legacy':
        _, trajectory = repair.prepare_trajectory(cp, frames=9, renderer_factory=SourceRenderer)
    else:
        _, trajectory = repair.prepare_saved_path(cp, renderer_factory=SourceRenderer)
    anchor = trajectory['anchors'][0]
    assert anchor['selection_rgb'] == cp.views[0]['original_rgb']
    assert preview.read_bytes() == before
    assert np.all(np.asarray(Image.open(cp.root / anchor['original_rgb'])) == 217)
    assert all(np.all(np.asarray(Image.open(cp.root / frame['rgb'])) == 217)
               for frame in trajectory['frames'])
    assert trajectory['recipe']['anchor_rgb_policy'] == 'viser'
    assert trajectory['frames'][anchor['frame_index']]['rgb'] == anchor['original_rgb']
    # Changing the generated anchor is detected on cache reuse, not silently accepted.
    (cp.root / anchor['original_rgb']).write_bytes(b'bad')
    with pytest.raises(ValueError, match='changed'):
        if kind == 'legacy':
            repair.prepare_trajectory(cp, frames=9)
        else:
            repair.prepare_saved_path(cp)
