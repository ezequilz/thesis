"""Execute SEVA → authors' VGGT oracle → original OracleGS optimizer."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from ..checkpoint import Checkpoint, atomic_json
from ..executor import run_process
from .inputs import export_inputs
from .adapter import known_model, assemble, validate_geometry, verify_sfm_cameras

HERE = Path(__file__).resolve().parent
DEFAULT_RUNTIME = {'repo':'/workspace/third_party/OracleGS',
    'seva_repo':'/workspace/third_party/oraclegs-seva', 'vggt_repo':'/workspace/third_party/oraclegs-vggt',
    'depth_repo':'/workspace/third_party/oraclegs-depth',
    'python':'/workspace/oraclegs-env/bin/python', 'model_python':'/workspace/oraclegs-model-env/bin/python',
    'colmap':'/workspace/oraclegs-model-env/bin/colmap',
    'depth_weights':'/workspace/models/oraclegs/depth_anything_v2_vitl.pth',
    'seva_fit':'crop', 'auto_setup':True, 'seva_resolution':'576p', 'oracle_adapter':'trajectory', 'oracle_target_frames':80,
    'oracle_path_order':None, 'iterations':22000, 'cap_max':1800000, 'seed':23}


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


def seva_request(inputs, output, manifest, runtime):
    width,height = manifest['resolution']
    request = {'data_path':str(inputs), 'data_items':'scene', 'task':'img2trajvid',
        'work_dir':str(output/'seva'), 'num_inputs':len(manifest['views']),
        'use_traj_prior':True, 'chunk_strategy':'interp-gt', 'chunk_strategy_first_pass':'gt-nearest',
        'cfg':[3.0,2.0], 'seed':runtime['seed'], 'W':width, 'H':height,
        'save_input':True, 'replace_or_include_input':False}
    # Leave T adaptive, as recommended for procedural two-pass img2trajvid.
    # 720p uses a disclosed padding adapter for the U-Net's 64-pixel stride.
    if manifest['seva_resolution'] == '576p':
        request['L_short'] = 576
    elif height % 64:
        request['pad_to_stride'] = 64
    return request


def training_args(dataset, model, n, runtime):
    return ['-s',str(dataset),'-m',str(model),'-r','1','--init_type','sfm',
            '--num_train_views',str(n),'--cap_max',str(runtime['cap_max']),
            '--iterations',str(runtime['iterations']),'--scale_reg','0.01','--opacity_reg','0.01',
            '--noise_lr','500000','--gt_synth_schedule','--gt_synth_ratio','1',
            '--lambda_lpips','0.3','--lambda_dssim_synth','0.3','--synth_attention_dir',
            '--depths','depths','--save_iterations',str(runtime['iterations'])]


def run_prepared(inputs, output_dir, *, runtime=None, should_stop=lambda:False, on_progress=lambda event:None):
    runtime = {**DEFAULT_RUNTIME, **(runtime or {})}
    inputs, output = Path(inputs).resolve(), Path(output_dir).resolve()
    output.mkdir(parents=True,exist_ok=True)
    if should_stop():
        raise InterruptedError('Cancelled before OracleGS setup')
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
    manifest = json.loads((inputs/'inputs.json').read_text())
    n = len(manifest['views'])
    dataset = output/'dataset'
    (dataset/'images').mkdir(parents=True)
    for view in manifest['views']:
        shutil.copyfile(inputs/view['image'],dataset/'images'/Path(view['image']).name)
    lock = json.loads((HERE/'upstream-lock.json').read_text())
    recipe = {'reconstruction_method':'oraclegs','inputs':manifest,
        'upstream':{k:{'url':v['url'],'revision':v['revision']} for k,v in lock.items()},
        'method_variant':'checkpoint-trajectory-adapter' if manifest['oracle_adapter']=='trajectory' else 'saved-poses-original-loader',
        'gt_policy':'Only saved reference RGBs are GT; all SEVA outputs are synthetic',
        'attention_layers':[0,22], 'attention_weights':[0.25,0.75],
        'schedule':'released train.py (differs from Beta schedule in paper)',
        'held_out_evaluation':'Unavailable: generated target images are not held-out GT',
        'runtime':runtime, 'commands':[]}
    atomic_json(output/'recipe.json',recipe)
    def stage(name, argv):
        if should_stop():
            raise InterruptedError('Cancelled before '+name)
        argv = [str(a) for a in argv]
        recipe['commands'].append({'phase':name,'argv':argv})
        atomic_json(output/'recipe.json',recipe)
        on_progress({'phase':name,'message':'OracleGS: '+name.replace('_',' ')})
        run_process(argv,output/(name+'.log'),should_stop)
    def model(name, action, repo_key, body, *, training=False):
        request = output/(name+'-request.json')
        atomic_json(request,body)
        python = Path(runtime['python' if training else 'model_python'])
        env = ['env','-u','PYTHONPATH','-u','PYTHONHOME','PYTHONNOUSERSITE=1',
               'LD_LIBRARY_PATH='+str(python.parent.parent/'lib')+':/usr/local/nvidia/lib:/usr/local/nvidia/lib64']
        stage(name,env+[python,'-u',HERE/'model_worker.py',action,runtime[repo_key],request])
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
    model('reference_depth','depth','depth_repo',{'images':str(dataset/'images'),'output':str(dataset/'depths'),
          'weights':runtime['depth_weights']})
    model('depth_alignment','script','repo',{'script':'utils/make_depth_scale.py','argv':[
          '--base_dir',str(dataset),'--depths_dir',str(dataset/'depths'),'--model_type','txt']},training=True)
    validate_geometry(output)
    model('seva_generation','seva','seva_repo',seva_request(inputs,output,manifest,runtime))
    assemble(inputs,output)
    attention = dataset/f'synthetic_{n}_attention_maps'
    model('vggt_attention','attention','vggt_repo',{'image_dir':str(output/'combined'),
          'output_dir':str(output/'attention-raw'),'num_images':None,'source_frame':0,
          'save_overlay':False,'seed':runtime['seed']})
    attention.mkdir()
    for image in (dataset/f'synthetic_{n}').glob('*.png'):
        source = output/'attention-raw'/('synth_'+image.name)
        if not source.is_file():
            raise ValueError(f'Missing attention map for {image.name}')
        shutil.copyfile(source,attention/image.name)
    args = training_args(dataset,output/'model',n,runtime)
    if manifest['oracle_adapter'] == 'trajectory':
        model('reconstruction','train','repo',{'references':str(dataset/'reference_names.json'),'argv':args},training=True)
    else:
        model('reconstruction','script','repo',{'script':'train.py','argv':args},training=True)
    native = output/f"model/point_cloud/iteration_{runtime['iterations']}/point_cloud.ply"
    if not native.is_file() or not native.stat().st_size:
        raise RuntimeError('Official OracleGS did not produce its final PLY')
    ply = output/'oraclegs.ply'
    shutil.copyfile(native,ply)
    result = {'reconstruction_method':'oraclegs','splat_path':str(ply),'mode':manifest['mode'],
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
    export_inputs(Checkpoint.load(checkpoint_dir),root/'inputs',mode,runtime=runtime,should_stop=should_stop)
    return run_prepared(root/'inputs',root/'results/oraclegs',runtime=runtime,
                        should_stop=should_stop,on_progress=on_progress)
