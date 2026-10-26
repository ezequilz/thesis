"""Execute SEVA → authors' VGGT oracle → original OracleGS optimizer."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from ..checkpoint import Checkpoint, atomic_json
from ..executor import run_process
from .adapter import known_model, assemble, validate_geometry, verify_sfm_cameras
from .provision import runtime_environment, huggingface_environment

HERE = Path(__file__).resolve().parent
DEFAULT_RUNTIME = {'repo':'/workspace/third_party/OracleGS',
    'seva_repo':'/workspace/third_party/oraclegs-seva', 'vggt_repo':'/workspace/third_party/oraclegs-vggt',
    'depth_repo':'/workspace/third_party/oraclegs-depth',
    'python':'/workspace/oraclegs-env/bin/python', 'model_python':'/workspace/oraclegs-model-env/bin/python',
    'colmap':'/workspace/oraclegs-model-env/bin/colmap',
    'depth_weights':'/workspace/models/oraclegs/depth_anything_v2_vitl.pth',
    'oracle_source_reference':'nearest', 'oracle_reference_mode':'scene_memory', 'oracle_preset':'spiral_expanding', 'oracle_preset_scale':0.25, 'oracle_split_mode':'single-split', 'seva_fit':'crop', 'auto_setup':True, 'seva_resolution':'576p', 'oracle_adapter':'trajectory', 'oracle_target_frames':240,
    'seva_geometry_guidance':'off', 'oracle_depth_source':'predicted', 'oracle_initialization':'sfm',
    'oracle_attention_batching':'within_trajectory', 'oracle_worldexplorer_attention':'weighted', 'oracle_path_order':None, 'oracle_iterations_testing':False, 'iterations':22000, 'cap_max':1800000, 'seed':23}


def validate_runtime(runtime):
    lock = json.loads((HERE/'upstream-lock.json').read_text())
    for key, field in [('oraclegs','repo'),('seva','seva_repo'),('vggt','vggt_repo'),('depth','depth_repo')]:
        repo = Path(runtime[field])
        if not repo.is_dir():
            raise FileNotFoundError(f'Missing {key} runtime: {repo}. Run oraclegs/provision.sh; see docs/oraclegs.md')
        revision = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
        if revision != lock[key]['revision']:
            raise ValueError(f'{key} revision differs from upstream-lock.json')
        for name, expected in lock[key]['files'].items():
            path = repo/name
            if not path.is_file():
                raise FileNotFoundError(f'Missing official source: {key}/{name}')
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f'Official source changed: {key}/{name}')
    for entry in lock['oraclegs']['submodules']:
        metadata, relative = entry.split('\t')
        if relative not in ('submodules/simple-knn','submodules/diff-gaussian-rasterization'):
            continue
        directory = Path(runtime['repo'])/relative
        if not (directory/'.git').exists():
            raise FileNotFoundError(f'Missing OracleGS CUDA submodule: {relative}')
        head = subprocess.check_output(['git','-C',str(directory),'rev-parse','HEAD'],text=True).strip()
        if head != metadata.split()[2]:
            raise ValueError(f'OracleGS CUDA submodule revision differs: {relative}')
    for key in ('python','model_python','colmap','depth_weights'):
        if not Path(runtime[key]).is_file():
            raise FileNotFoundError(f'Missing OracleGS {key}: {runtime[key]}')
    for key in ('iterations','cap_max'):
        if type(runtime[key]) is not int or runtime[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    probe_runtime(runtime)


def probe_runtime(runtime):
    """Existing executables are not proof that a failed setup completed."""
    checks = {
        'python': 'import torch, diff_gaussian_rasterization, simple_knn._C; '
                  'assert torch.__version__ == "1.13.1+cu117", torch.__version__',
        'model_python': 'import torch; import seva.eval; '
                        'from seva.modules.autoencoder import AutoEncoder; '
                        'from seva.modules.conditioner import CLIPConditioner; '
                        'assert torch.__version__.split("+")[0] == "2.5.1", torch.__version__',
    }
    for key, code in checks.items():
        python = Path(runtime[key])
        try:
            subprocess.check_output([str(python), '-c', code],
                env=runtime_environment(python.parent.parent),
                stderr=subprocess.STDOUT, text=True, timeout=60)
        except (OSError, subprocess.CalledProcessError) as exc:
            detail = getattr(exc, 'output', '') or str(exc)
            raise FileNotFoundError(f'Incomplete OracleGS {key} runtime: {detail[-4000:]}') from exc


def validate_staged_trajectory(inputs, manifest):
    """Do not silently reuse a legacy total-budget path for a new GPU run."""
    if manifest.get('oracle_adapter') == 'per_view_preset':
        from .presets import validate_preset_staging
        validate_preset_staging(inputs, manifest)
        return
    if manifest['oracle_adapter'] != 'trajectory':
        return
    trajectory = manifest.get('trajectory', {})
    if (trajectory.get('version') not in (5, 7) or
            trajectory.get('frame_count_semantics') != 'per-segment-interior-targets'):
        raise ValueError('Legacy OracleGS trajectory: restart the queue manager, then restage a fresh run for independent pairwise generation')
    order = manifest['oracle_path_order']
    expected = (len(order)-1)*manifest['oracle_target_frames']
    n = len(manifest['views'])
    split = json.loads((Path(inputs)/'scene'/f'train_test_split_{n}.json').read_text())
    frames = json.loads((Path(inputs)/'scene/transforms.json').read_text())['frames']
    if (trajectory.get('target_count') != expected or len(trajectory.get('frames', [])) != expected
            or len(frames) != n+expected or split['train_ids'] != list(range(n))
            or split['test_ids'] != list(range(n,n+expected))):
        raise ValueError('OracleGS staged trajectory count/split disagrees with frames per transition')
    if trajectory.get('version') == 7:
        from .splitting import validate_split_staging
        validate_split_staging(inputs, manifest)
        return
    if manifest.get('seva_task') != 'img2vid':
        raise ValueError('Pairwise trajectories require ordered SEVA img2vid')
    segments = manifest.get('seva_segments', [])
    if len(segments) != len(order)-1:
        raise ValueError('OracleGS needs one SEVA run per camera pair')
    for j, segment in enumerate(segments):
        targets = list(range(j*manifest['oracle_target_frames'], (j+1)*manifest['oracle_target_frames']))
        if (segment['index'] != j or segment['input_subdir'] != f'segments/{j:03d}'
                or segment['target_indices'] != targets
                or [segment['from_reference'],segment['to_reference']] != order[j:j+2]):
            raise ValueError('Invalid SEVA segment mapping')
        native = Path(inputs)/segment['input_subdir']/'scene'
        a,b = order[j:j+2]
        frame_order = [i for i in range(n) if i not in (a,b)]+[a]+[n+i for i in targets]+[b]
        selected = [frames[i] for i in frame_order]
        if (segment.get('frame_order') != frame_order or
                segment.get('endpoint_frame_indices') != [n-2,len(selected)-1]):
            raise ValueError('Invalid SEVA segment endpoint/frame mapping')
        split = {'train_ids':[i for i,k in enumerate(frame_order) if k < n],
                 'test_ids':[i for i,k in enumerate(frame_order) if k >= n]}
        if (json.loads((native/'transforms.json').read_text())['frames'] != selected or
                json.loads((native/f'train_test_split_{n}.json').read_text()) != split):
            raise ValueError('SEVA segment images/cameras disagree with the staged trajectory')



def seva_request(inputs, output, manifest, runtime, segment=None):
    width,height = manifest['resolution']
    task = manifest.get('seva_task', 'img2trajvid')
    request = {'data_path':str(inputs), 'data_items':'scene', 'task':task,
        'work_dir':str(output/'seva'), 'num_inputs':len(manifest['views']),
        'use_traj_prior':True, 'chunk_strategy':'interp-gt', 'chunk_strategy_first_pass':'gt-nearest',
        'cfg':[3.0,2.0], 'seed':runtime['seed'], 'W':width, 'H':height,
        'save_first_pass':True, 'save_input':True, 'replace_or_include_input':task == 'img2vid'}
    if manifest.get('seva_segments') and task == 'img2vid':
        request['ordered_pair_endpoints'] = [len(manifest['views'])-2, len(manifest['views'])+manifest['oracle_target_frames']-1]
    if manifest.get('seva_geometry_guidance', 'off') == 'source_splat':
        asset_root = Path(inputs).parents[1] if manifest.get('seva_segments') else Path(inputs)
        request['source_geometry_guidance'] = {'inputs':str(asset_root), 'strength':0.35}
    if manifest.get('oracle_adapter') == 'per_view_preset':
        preset = segment['preset'] if segment is not None else manifest['oracle_preset']
        request.update(num_inputs=segment.get('reference_count', 1) if segment else len(manifest['views']), camera_scale=2.0, guider_types=[1,2], replace_or_include_input=True,
                       cfg=[4.0 if preset == 'orbit' else 3.0, 2.0])
    if segment is not None:
        request['num_inputs'] = segment.get('reference_count', len(manifest['views']))
        request['replace_or_include_input'] = True
        if request['num_inputs'] == 1 and 'source_geometry_guidance' in request:
            request['source_geometry_guidance']['source_reference'] = segment['from_reference']
    if (manifest.get('oracle_adapter') == 'per_view_preset'
            and manifest.get('oracle_source_reference', 'nearest') == 'fixed_initial'):
        if segment is None:
            raise ValueError('Fixed initial source requires a staged preset clip')
        source = segment['from_reference']
        request['fixed_initial_source'] = {
            'frame_index':segment['frame_order'].index(source),
            'reference_index':source, 'mode':manifest['mode']}
        if 'source_geometry_guidance' in request:
            request['source_geometry_guidance']['source_reference'] = source
    if manifest.get('oracle_reference_mode') == 'single':
        if request['num_inputs'] != 1:
            raise ValueError('Single-image SEVA conditioning requires exactly one staged reference per trajectory')
        # Match the native single-image context: one reference + 20 anchors.
        # num_prior_frames is an upstream minimum; long paths may need more
        # anchors, sampled in multiple chunks with the same 21-frame context.
        request.update(T=21, num_prior_frames=20)
    if manifest.get('oracle_reference_mode') == 'scene_memory':
        request.update(T=[27, 21], chunk_strategy_first_pass='gt')
    # 720p uses a disclosed padding adapter for the U-Net's 64-pixel stride.
    if manifest['seva_resolution'] == '576p':
        request['L_short'] = 576
    elif height % 64:
        request['pad_to_stride'] = 64
    return request


def save_iterations(runtime):
    final = runtime['iterations']
    scheduled = range(1000, final + 1, 1000) if runtime.get('oracle_iterations_testing') else (3000, 7000, 15000)
    return sorted({i for i in scheduled if i <= final} | {final})


def training_args(dataset, model, n, runtime):
    return ['-s',str(dataset),'-m',str(model),'-r','1','--init_type','sfm',
            '--num_train_views',str(n),'--cap_max',str(runtime['cap_max']),
            '--iterations',str(runtime['iterations']),'--scale_reg','0.01','--opacity_reg','0.01',
            '--noise_lr','500000','--gt_synth_schedule','--gt_synth_ratio','1',
            '--lambda_lpips','0.3','--lambda_dssim_synth','0.3','--synth_attention_dir',
            '--depths','depths','--save_iterations',
            *map(str, save_iterations(runtime))]


def run_prepared(inputs, output_dir, *, runtime=None, should_stop=lambda:False, on_progress=lambda event:None):
    runtime = {**DEFAULT_RUNTIME, **(runtime or {})}
    inputs, output = Path(inputs).resolve(), Path(output_dir).resolve()
    output.mkdir(parents=True,exist_ok=True)
    if should_stop():
        raise InterruptedError('Cancelled before OracleGS setup')
    manifest = json.loads((inputs/'inputs.json').read_text())
    from .backend import validate_options
    validated = validate_options(runtime)
    for key in ('oracle_source_reference', 'oracle_reference_mode', 'oracle_split_mode'):
        if key in manifest and manifest[key] != validated[key]:
            raise ValueError(key + ' changed after staging; restage inputs')
    for key, default in [('seva_geometry_guidance', 'off'), ('oracle_depth_source', 'predicted'), ('oracle_initialization', 'sfm')]:
        if manifest.get(key, default) != runtime.get(key, default):
            raise ValueError(key + ' changed after staging; restage inputs')
    from .source_geometry import validate_assets
    validate_assets(inputs, manifest)
    validate_staged_trajectory(inputs, manifest)
    try:
        validate_runtime(runtime)
    except FileNotFoundError:
        if not runtime.get('auto_setup'):
            raise
        base = Path(runtime['repo']).parent.parent
        for key in ('repo','seva_repo','vggt_repo','depth_repo','python','model_python','colmap','depth_weights'):
            expected = base / Path(DEFAULT_RUNTIME[key]).relative_to('/workspace')
            if Path(runtime[key]) != expected:
                raise ValueError('Automatic OracleGS setup requires the standard directory layout; provision custom runtimes explicitly')
        on_progress({'phase':'setup','message':'Installing isolated OracleGS, SEVA, VGGT and depth runtimes'})
        run_process(['bash',str(HERE/'provision.sh'),str(base)],output/'setup.log',should_stop)
        validate_runtime(runtime)
    if (output/'dataset').exists():
        raise ValueError('OracleGS output already contains a dataset; use a fresh isolated run')
    if should_stop():
        raise InterruptedError('Cancelled after OracleGS setup')
    n = len(manifest['views'])
    dataset = output/'dataset'
    (dataset/'images').mkdir(parents=True)
    for view in manifest['views']:
        if should_stop():
            raise InterruptedError('Cancelled while staging OracleGS references')
        shutil.copyfile(inputs/view['image'],dataset/'images'/Path(view['image']).name)
    lock = json.loads((HERE/'upstream-lock.json').read_text())
    recipe = {'reconstruction_method':'oraclegs','inputs':manifest,
        'upstream':{k:{'url':v['url'],'revision':v['revision']} for k,v in lock.items()},
        'method_variant':('per-view-seva-'+manifest['oracle_preset'] if manifest.get('oracle_adapter') == 'per_view_preset' else 'independent-pairwise-trajectories' if manifest.get('seva_segments') else
                          'checkpoint-trajectory-adapter' if manifest['oracle_adapter']=='trajectory' else 'saved-poses-original-loader'),
        'gt_policy':'Only saved reference RGBs are GT; all SEVA outputs are synthetic',
        'attention_layers':[0,22], 'attention_weights':[0.25,0.75],
        'attention_policy':'all-references-nearest-segment-endpoint-v1',
        'attention_batching':runtime['oracle_attention_batching'],
        'worldexplorer_attention':manifest.get('oracle_worldexplorer_attention', 'weighted'),
        'schedule':'released train.py (differs from Beta schedule in paper)',
        'save_iterations':save_iterations(runtime),
        'held_out_evaluation':'Unavailable: generated target images are not held-out GT',
        'graceful_stop':'In-memory stop-only hook after completed optimizer/MCMC iteration; atomic PLY export; 240s timeout',
        'runtime':runtime, 'commands':[]}
    if manifest.get('artifixer_source'):
        recipe.update(method_variant='artifixer-generated-images', gt_policy='Six GPT repaired references are GT; all ArtiFixer outputs are synthetic', artifixer_source=manifest['artifixer_source'])
    atomic_json(output/'recipe.json',recipe)
    def stage(name, argv, *, training=False):
        if should_stop():
            raise InterruptedError('Cancelled before '+name)
        argv = [str(a) for a in argv]
        recipe['commands'].append({'phase':name,'argv':argv})
        atomic_json(output/'recipe.json',recipe)
        on_progress({'phase':'seva_generation' if name.startswith('seva_generation_') else name,
                     'log_phase':name, 'message':'OracleGS: '+name.replace('_',' ')})
        poll = should_stop
        if training:
            from .graceful import stop_poll
            poll = stop_poll(output, should_stop, on_progress)
        try:
            run_process(argv,output/(name+'.log'),poll)
        except Exception:
            if training and (output/'SAVE_STOP').exists():
                raise InterruptedError('OracleGS stopped; recovering the saved PLY (or last confirmed snapshot)')
            raise
        if training and ((output/'SAVE_STOP').exists() or should_stop()):
            raise InterruptedError('OracleGS stopped early; partial reconstruction saved')
    def model(name, action, repo_key, body, *, training=False):
        request = output/(name+'-request.json')
        if training:
            body = {**body, 'stop_control':str(output/'SAVE_STOP')}
        atomic_json(request,body)
        python = Path(runtime['python' if training else 'model_python'])
        env = ['env','-u','PYTHONPATH','-u','PYTHONHOME','PYTHONNOUSERSITE=1',
               'LD_LIBRARY_PATH='+str(python.parent.parent/'lib')+':/usr/local/nvidia/lib:/usr/local/nvidia/lib64']
        env += [key+'='+value for key,value in huggingface_environment().items()]
        stage(name,env+[python,'-u',HERE/'model_worker.py',action,runtime[repo_key],request],training=training)
    if not manifest.get('artifixer_source'):
        model('model_access','assets','seva_repo',{'output':str(output/'vae-provenance.json')})
    sparse = dataset/'sparse/0'
    source_only = (manifest.get('oracle_depth_source') == 'source_splat' and
                   manifest.get('oracle_initialization') in ('source_splat', 'point_cloud'))
    if source_only:
        from .adapter import source_camera_model
        source_camera_model(inputs, output)
    else:
        colmap = runtime['colmap']
        db = output/'database.db'
        stage('sfm_features',[colmap,'feature_extractor','--database_path',db,'--image_path',dataset/'images',
              '--ImageReader.camera_model','PINHOLE','--ImageReader.single_camera_per_image','1'])
        known_model(inputs,output)
        stage('sfm_matching',[colmap,'exhaustive_matcher','--database_path',db])
        sparse = dataset/'sparse/0'
        sparse.mkdir(parents=True)
        stage('sfm_triangulation',[colmap,'point_triangulator','--database_path',db,'--image_path',dataset/'images',
              '--input_path',output/'known','--output_path',sparse,
              '--Mapper.ba_refine_focal_length','0','--Mapper.ba_refine_principal_point','0',
              '--Mapper.ba_refine_extra_params','0','--Mapper.fix_existing_images','1',
              '--Mapper.tri_ignore_two_view_tracks','0','--refine_intrinsics','0'])
        stage('sfm_export',[colmap,'model_converter','--input_path',sparse,'--output_path',sparse,'--output_type','TXT'])
    verify_sfm_cameras(inputs,output)
    if manifest.get('oracle_depth_source', 'predicted') == 'predicted':
        model('reference_depth','depth','depth_repo',{'images':str(dataset/'images'),'output':str(dataset/'depths'),
              'weights':runtime['depth_weights']})
        model('depth_alignment','script','repo',{'script':'utils/make_depth_scale.py','argv':[
              '--base_dir',str(dataset),'--depths_dir',str(dataset/'depths'),'--model_type','txt']},training=True)
        validate_geometry(output)
    else:
        points = (sparse/'points3D.txt').read_text().splitlines()
        point_count = sum(bool(line.strip()) and not line.startswith('#') for line in points)
        if manifest.get('oracle_initialization', 'sfm') == 'sfm' and not point_count:
            raise ValueError('Sparse triangulation produced no points for SfM initialization')
        atomic_json(output/'geometry-check.json', {
            'triangulated_points':point_count,
            'passed':True, 'depth_source':'source_splat', 'colmap_skipped':source_only,
            'scope':'Validated staged camera-Z inverse depth and masks; no monocular prediction or SfM depth alignment',
            'views':manifest['source_depth']['views']})
    if manifest.get('artifixer_source'):
        from .artifixer_inputs import publish_generated
        publish_generated(inputs, output, manifest, should_stop=should_stop)
    elif manifest.get('seva_segments'):
        from .segments import merge_segments
        for segment in manifest['seva_segments']:
            index = segment['index']
            destination = output/'seva-segments'/f'{index:03d}'
            destination.mkdir(parents=True)
            request = seva_request(inputs/segment['input_subdir'], destination, manifest, runtime, segment=segment)
            if manifest.get('oracle_reference_mode') == 'scene_memory':
                from .scene_memory import prepare_memory
                request['scene_memory'] = prepare_memory(inputs, output, manifest, segment, destination)
            model(f'seva_generation_{index+1:03d}_of_{len(manifest["seva_segments"]):03d}',
                  'seva','seva_repo',request)
        merge_segments(inputs, output, manifest, should_stop=should_stop)
    else:
        request = seva_request(inputs,output,manifest,runtime)
        if manifest.get('oracle_reference_mode') == 'scene_memory':
            from .scene_memory import prepare_memory
            request['scene_memory'] = prepare_memory(inputs, output, manifest, None, output)
        model('seva_generation','seva','seva_repo',request)
    assemble(inputs,output,should_stop=should_stop)
    attention = dataset/f'synthetic_{n}_attention_maps'
    from .adapter import append_worldexplorer
    training_n, original_names = append_worldexplorer(inputs, output, n, should_stop=should_stop)
    weighted_world = manifest.get('oracle_worldexplorer_attention', 'weighted') == 'weighted'
    from .attention import plan_chunks, worldexplorer_targets
    extra = worldexplorer_targets(inputs, output, manifest) if weighted_world and original_names else []
    chunks = plan_chunks(manifest, extra_targets=extra, mode=runtime['oracle_attention_batching'], seed=runtime['seed'])
    atomic_json(output/'attention-chunks.json', {'policy':recipe['attention_policy'], 'worldexplorer_attention':recipe['worldexplorer_attention'], 'worldexplorer_source_policy':'nearest-camera-center-direction-tiebreak', 'batching':runtime['oracle_attention_batching'], 'seed':runtime['seed'], 'chunks':chunks})
    model('vggt_attention','attention','vggt_repo',{'image_dir':str(output/'combined'),
          'output_dir':str(output/'attention-raw'), 'chunks':chunks, 'seed':runtime['seed']})
    attention.mkdir()
    for image in (dataset/f'synthetic_{n}').glob('*.png'):
        if not weighted_world and image.name in original_names:
            continue
        source = output/'attention-raw'/('synth_'+image.name)
        if not source.is_file():
            raise ValueError(f'Missing attention map for {image.name}')
        shutil.copyfile(source,attention/image.name)
    args = training_args(dataset,output/'model',training_n,runtime)
    geometry_request = {'oracle_initialization':manifest.get('oracle_initialization', 'sfm'), 'unweighted_synthetic': [] if weighted_world else original_names, 'source_geometry_report':str(output/'source-geometry-runtime.json'),
                        'source_depth':str(inputs/'source-depth') if manifest.get('oracle_depth_source') == 'source_splat' else None,
                        'source_initialization':str(inputs/'source-initialization.ply') if manifest.get('source_initialization') else None}
    if manifest.get('oracle_depth_source') == 'source_splat':
        args[args.index('--depths')+1] = ''
    if manifest.get('oracle_initialization') == 'source_splat':
        args += ['--sh_degree', str(manifest['source_initialization']['sh_degree'])]
    if manifest.get('artifixer_source') or original_names or manifest['oracle_adapter'] in ('trajectory', 'per_view_preset'):
        model('reconstruction','train','repo',{'references':str(dataset/'reference_names.json'),'argv':args, **geometry_request},training=True)
    else:
        model('reconstruction','script','repo',{'script':'train.py','argv':args, **geometry_request},training=True)
    native = output/f"model/point_cloud/iteration_{runtime['iterations']}/point_cloud.ply"
    if not native.is_file() or not native.stat().st_size:
        raise RuntimeError('Official OracleGS did not produce its final PLY')
    ply = output/'oraclegs.ply'
    shutil.copyfile(native,ply)
    result = {'reconstruction_method':'oraclegs','splat_path':str(ply),'mode':manifest['mode'],
              'saved_iteration':runtime['iterations'],
              'method_variant':recipe['method_variant'],'input_manifest':str(inputs/'inputs.json'),
              'recipe':str(output/'recipe.json'),'camera_alignment':str(output/'camera-alignment.json'),'output_dir':str(output)}
    atomic_json(output/'result.json',result)
    return result


def run_repair(checkpoint_dir, output_dir, *, mode='edited', runtime=None,
               should_stop=lambda:False,on_progress=lambda event:None, seed=None, **unused):
    runtime = {**DEFAULT_RUNTIME, **(runtime or {})}
    if seed is not None:
        runtime['seed'] = seed
    root = Path(output_dir).resolve()
    root.mkdir(parents=True,exist_ok=True)
    from .inputs import export_inputs
    export_inputs(Checkpoint.load(checkpoint_dir),root/'inputs',mode,runtime=runtime,should_stop=should_stop)
    try:
        return run_prepared(root/'inputs',root/'results/oraclegs',runtime=runtime,
                            should_stop=should_stop,on_progress=on_progress)
    except Exception as exc:
        from .partial import salvage
        try:
            salvage(root/'results/oraclegs', str(exc))
        except Exception as recovery_error:
            atomic_json(root/'snapshot-recovery-error.json', {'message':str(recovery_error)})
        raise
