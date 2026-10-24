"""Offline tests of request construction; no browser, services or paid calls."""
import shutil
import subprocess
from pathlib import Path

import pytest


def test_dashboard_stage_payloads_and_frozen_schedule_selection():
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node is required for the dashboard JavaScript logic check')
    page = Path(__file__).resolve().parents[1] / 'src/splat_explorer/web/static/splatfix.html'
    code = r'''
const fs=require('fs'), vm=require('vm'), assert=require('assert/strict');
const source=fs.readFileSync(process.argv[1],'utf8').split('<script>')[1].split('</script>')[0];
const elements=new Map();
const get=id=>{if(!elements.has(id))elements.set(id,{value:id==='artifixer-max-gaussians'?'1000000':id==='handoff-add-noise'?'repaired_pixel_sprinkle':'',textContent:'',innerHTML:'',disabled:false,valid:true,opened:false,
  selectedOptions:[{textContent:'Other scene'}],classList:{toggle(){}},style:{},addEventListener(){},querySelectorAll(){return[]},
  setCustomValidity(message){this.valid=!message},reportValidity(){return this.valid},showModal(){this.opened=true},close(){this.opened=false}});return elements.get(id)};
const context=vm.createContext({document:{getElementById:get,querySelectorAll(){return [{id:"future-setting",name:"",type:"select-one",tagName:"SELECT",value:"future",labels:[{textContent:"Future setting"}],selectedOptions:[{value:"future",textContent:"Future option"}],matches(){return false},getClientRects(){return [{}]}}]}},fetch:()=>new Promise(()=>{}),setInterval(){},setTimeout(){},assert});
vm.runInContext(source,context);
Promise.resolve(vm.runInContext(`(async()=>{
  state={scenes:[],jobs:[],checkpoints:[{id:'/saved/cp',name:'Saved camera checkpoint',scene:'Original scene',complete:true,
    target_views:7,edited:7,rgb_renderer:'viser',trajectory_caches:1,reconstruction_ready:true,views:[]}]};
  $('reconstruction-method').value='artifixer';
  selected='/saved/cp';$('scene').value='unrelated-current-scene';$('views').value='7';$('mode').value='edited';
  assert.equal(dashboardSnapshot().controls[0].selected_options[0].text,'Future option');
  assert.equal(dashboardSnapshot().controls[0].label,'Future setting');
  assert.equal(payload('select').views,7);
  assert.equal(payload('select').scene_id,'unrelated-current-scene');
  assert.equal(payload('repair').checkpoint,'/saved/cp');
  assert.equal(payload('repair').mode,'edited');
  assert.equal(payload('repair').handoff_generation,false);
  assert.equal(payload('repair').handoff_backward_pass,true);
  $('handoff-backward-pass').value='off';
  assert.equal(payload('repair').handoff_backward_pass,false);
  $('handoff-backward-pass').value='on';
  assert.equal(payload('repair').gpt_image_context,'closest_repaired_image');
  $('gpt-image-context').value='original_rgb_capture';
  assert.equal(payload('repair').gpt_image_context,'original_rgb_capture');
  $('gpt-image-context').value='closest_repaired_image';
  assert.equal(payload('repair').handoff_add_noise,true);
  assert.equal(payload('repair').handoff_noise_strength,25);
  assert.equal(payload('repair').handoff_noise_mode,'repaired_pixel_sprinkle');
  $('handoff-add-noise').value='on';
  assert.equal(payload('repair').handoff_noise_mode,'gaussian');
  $('handoff-add-noise').value='repaired_pixel_sprinkle';
  assert.equal(payload('repair').handoff_noise_mode,'repaired_pixel_sprinkle');
  assert.equal(payload('repair').handoff_add_noise,true);
  assert.equal(payload('repair').handoff_noise_vignette,true);
  $('handoff-noise-vignette').value='off';
  assert.equal(payload('repair').handoff_noise_vignette,false);
  $('handoff-add-noise').value='off';
  $('handoff-noise-strength').value='12';
  assert.equal(payload('repair').handoff_add_noise,false);
  assert.equal(payload('repair').handoff_noise_strength,12);
  assert.equal(payload('benchmark').handoff_add_noise,undefined);
  $('gpt-image-context').value='closest_repaired_image';
  assert.equal(payload('repair').gpt_image_context,'closest_repaired_image');
  $('gpt-image-context').value='initial_repaired_image';
  assert.equal(payload('repair').gpt_image_context,'initial_repaired_image');
  assert.equal(payload('benchmark').gpt_image_context,undefined);
  $('handoff-generation').value='on';
  assert.equal(payload('repair').handoff_generation,true);
  assert.equal(payload('repair').split_mode,'single-split');
  assert.equal(payload('benchmark').handoff_generation,undefined);
  $('reconstruction-method').value='oraclegs';
  assert.equal(payload('repair').handoff_generation,undefined);
  $('reconstruction-method').value='artifixer';
  $('handoff-generation').value='off';
  for(const pipeline of ['authors_3dgut','authors_3dgut_viser_rgb','legacy_gsplat']){
    $('artifixer-pipeline').value=pipeline;
    for(const rgb of ['on','off'])for(const downscale of ['on','off']){
      $('artifixer-generated-rgb-loss').value=rgb;
      $('artifixer-generated-lpips-downscale').value=downscale;
      renderArtifixerPipeline();
      assert.equal($('artifixer-generated-rgb-loss').disabled,false);
      assert.equal($('artifixer-generated-lpips-downscale').disabled,false);
      assert.equal(payload('repair').generated_rgb_loss,rgb==='on');
      assert.equal(payload('repair').generated_lpips_downscale,downscale==='on');
      assert.equal(payload('benchmark').generated_rgb_loss,undefined);
    }
  }
  // Endpoint selection must not leave legacy checkpoints in no-split mode.
  for(const split of ['double-split','single-split','no-split']){
    $('split-mode').value=split;
    updateEndpointControls(false);
    assert.equal($('endpoint-reference-mode').disabled,true);
    updateEndpointControls(true);
    updateEndpointControls(true); // Rendering twice must not overwrite the saved choice.
    assert.equal($('split-mode').value,'no-split');
    assert.equal($('split-mode').disabled,true);
    assert.equal($('endpoint-reference-mode').disabled,false);
    updateEndpointControls(false);
    assert.equal(payload('repair').split_mode,split);
    assert.equal($('split-mode').disabled,false);
  }
  $('split-mode').value='double-split';
  $('artifixer-pipeline').value='authors_3dgut';renderArtifixerPipeline();
  assert.equal(payload('repair').artifixer_pipeline,'authors_3dgut');
  assert.equal($('artifixer-reconstruction-initialization').disabled,true);
  assert.equal($('artifixer-reconstruction-initialization').value,'point_cloud');
  $('artifixer-pipeline').value='authors_3dgut_viser_rgb';renderArtifixerPipeline();
  assert.equal(payload('repair').artifixer_pipeline,'authors_3dgut_viser_rgb');
  assert.equal($('artifixer-reconstruction-initialization').disabled,true);
  $('artifixer-pipeline').value='legacy_gsplat';renderArtifixerPipeline();
  assert.equal(payload('repair').artifixer_pipeline,'legacy_gsplat');
  assert.equal($('artifixer-reconstruction-initialization').disabled,false);
  $('artifixer-reconstruction-images').value='all';
  $('artifixer-reconstruction-initialization').value='full_splat';
  assert.equal(payload('repair').reconstruction_images,'all');
  assert.equal(payload('repair').reconstruction_initialization,'full_splat');
  $('artifixer-reconstruction-images').value='distilled';
  $('artifixer-reconstruction-initialization').value='point_cloud';
  assert.equal(payload('repair').reconstruction_images,'distilled');
  assert.equal(payload('repair').reconstruction_initialization,'point_cloud');
  assert.equal(payload('benchmark').reconstruction_images,undefined);
  $('image-cache-insertion').value='seed';
  assert.equal(payload('repair').initial_image_seeding,true);
  assert.equal(payload('repair').image_cache_insertion,false);
  $('image-cache-insertion').value='off';
  assert.equal(payload('repair').initial_image_seeding,false);
  $('image-cache-insertion').value='on';
  assert.equal(payload('repair').image_cache_insertion,true);
  assert.equal(payload('benchmark').image_cache_insertion,undefined);
  $('benchmark-runs').value='3';
  assert.equal(payload('benchmark').benchmark_runs,3);
  schedule('benchmark');
  assert.equal(pendingPayload.benchmark_runs,3);
  assert.ok($('schedule-context').textContent.includes('3 consecutive run(s)'));
  $('benchmark-runs').value='1';
  assert.equal(pendingPayload.benchmark_runs,3);
  $('benchmark-runs').valid=false;
  pendingStage='';schedule('benchmark');
  assert.equal(pendingStage,'');
  $('benchmark-runs').valid=true;

  $('reconstruction-method').value='oraclegs';
  $('oracle-worldexplorer-attention').value='weighted';
  $('oracle-attention-batching').value='global';
  assert.equal(payload('repair').oracle_worldexplorer_attention,'weighted');
  assert.equal(payload('repair').oracle_attention_batching,'global');
  $('oracle-worldexplorer-attention').value='unweighted';
  assert.equal(payload('repair').oracle_worldexplorer_attention,'unweighted');
  assert.equal(payload('repair').oracle_source_reference,'nearest');
  $('oracle-source-reference').value='fixed_initial';
  assert.equal(payload('repair').oracle_source_reference,'fixed_initial');
  $('oracle-source-reference').value='nearest';
  $('oracle-reference-mode').value='scene_memory';
  assert.equal(payload('repair').oracle_reference_mode,'scene_memory');
  $('oracle-reference-mode').value='all';
  $('oracle-split-mode').value='single-split';
  updateOracleReferenceControls();
  assert.equal(payload('repair').oracle_reference_mode,'all');
  $('oracle-reference-mode').value='single';updateOracleReferenceControls();
  assert.equal($('oracle-split-mode').disabled,true);
  assert.equal(payload('repair').oracle_split_mode,'double-split');
  $('oracle-reference-mode').value='all';updateOracleReferenceControls();
  assert.equal($('oracle-split-mode').disabled,false);
  assert.equal($('oracle-split-mode').value,'single-split');
  $('oracle-adapter').value='per_view_preset';
  $('oracle-preset').value='shuffle_orbit';
  $('oracle-preset-scale').value='0.25';
  updatePresetControls();
  assert.equal($('oracle-preset-controls').hidden,false);
  assert.equal(payload('repair').oracle_preset,'shuffle_orbit');
  assert.equal(payload('repair').oracle_preset_scale,0.25);
  schedule('repair');
  $('oracle-preset').value='orbit';
  assert.equal(pendingPayload.oracle_preset,'shuffle_orbit');
  $('oracle-adapter').value='trajectory';updatePresetControls();
  assert.equal($('oracle-preset-controls').hidden,true);
  $('reconstruction-method').value='g4splat';
  $('g4splat-source-depth').value='off';
  assert.equal(payload('repair').source_depth,false);
  $('g4splat-source-depth').value='on';
  assert.equal(payload('repair').source_depth,true);
  assert.equal(payload('repair').regularization_profile,undefined);
  schedule('repair');
  $('g4splat-source-depth').value='off';
  assert.equal(pendingPayload.source_depth,true);
  state.g4splat_baselines=[{id:'scene1',label:'Room',ready:false,status:'Awaiting assets',reason:'Missing original',protocol:'Untouched inputs'}];
  refreshG4SplatBaselines();
  assert.equal($('g4splat-baseline-scene').value,'scene1');
  assert.equal($('g4splat-baseline-now').disabled,true);
  assert.match($('g4splat-baseline-status').textContent,/Missing original/);
  $('reconstruction-method').value='artifixer';
  assert.equal(payload('repair').source_depth,undefined);
  assert.equal(payload('repair').reconstruction_method,'artifixer');
  assert.equal(payload('repair').split_mode,'double-split');
  $('split-mode').value='single-split';
  assert.equal(payload('repair').split_mode,'single-split');
  $('regularization-profile').value='base_mcmc';
  assert.equal(payload('repair').regularization_profile,'base_mcmc');
  assert.equal(payload('repair').scene_id,undefined);
  assert.equal(payload('edit').scene_id,undefined);
  assert.equal(payload('edit').endpoint_repair_context,'initial_repair');
  $('endpoint-repair-context').value='prior_repair';
  assert.equal(payload('edit').endpoint_repair_context,'prior_repair');
  $('endpoint-repair-context').value='rgb_only';
  assert.equal(payload('edit').endpoint_repair_context,'rgb_only');
  schedule('repair');
  assert.equal($('schedule-dialog').opened,true);
  assert.match($('schedule-context').textContent,/Saved camera checkpoint/);
  assert.match($('schedule-context').textContent,/GPT-image improved/);
  const saved=pendingPayload;
  $('artifixer-reconstruction-images').value='all';
  $('artifixer-reconstruction-initialization').value='full_splat';
  $('reconstruction-method').value='other-method';
  selected='/different-checkpoint';$('mode').value='baseline';$('regularization-profile').value='artifixer';
  let submitted;
  api=async(path,body)=>{submitted=body;return {}};refresh=async()=>{};
  await queue('repair','2099-01-01T00:00:00Z',saved);
  assert.equal(submitted.checkpoint,'/saved/cp');
  assert.equal(submitted.mode,'edited');
  assert.equal(submitted.image_cache_insertion,true);
  assert.equal(submitted.reconstruction_images,'distilled');
  assert.equal(submitted.reconstruction_initialization,'point_cloud');
  assert.equal(submitted.reconstruction_method,'artifixer');
  $('reconstruction-method').value='artifixer';
  assert.equal(submitted.regularization_profile,'base_mcmc');
  assert.equal(submitted.split_mode,'single-split');
  assert.equal(submitted.scheduled_at,'2099-01-01T00:00:00Z');
  $('views').valid=false;$('schedule-dialog').opened=false;
  schedule('select');
  assert.equal($('schedule-dialog').opened,false);
  // Custom uses the selected scene and dimensions without queueing a VLM job.
  $('views').valid=true;$('views').value='4';$('resolution-profile').value='training';
  $('scene-capture').scrollIntoView=()=>{};
  let customPath;
  api=async(path,body)=>{customPath=path;submitted=body;return {checkpoint:'/custom/saved'}};
  await startCustom();
  assert.equal(customPath,'/custom');assert.equal(submitted.views,4);
  assert.equal(submitted.scene_id,'unrelated-current-scene');
  assert.equal(submitted.resolution_profile,'training');assert.equal(selected,'/custom/saved');
  api=async(path,body)=>{submitted=body;return {}};
  // Baseline is available before editing, while improved waits for every repair.
  selected='/saved/cp';const cp=state.checkpoints[0];cp.views=[{camera:{width:16,height:16},original_url:'/original.png',metadata:{selected:true},image_repair:{status:'failed',error:'retry me'}}];cp.edited=0;
  $('mode').value='baseline';renderCheckpoint();
  assert.equal($('repair-now').disabled,false);
  assert.equal($('repair-schedule').disabled,false);
  assert.match($('checkpoint-facts').innerHTML,/Baseline: ready/);
  assert.match($('view-grid').innerHTML,/Saved camera & image metadata/);
  assert.match($('view-grid').innerHTML,/retry me/);
  $('mode').value='edited';renderCheckpoint();
  assert.equal($('repair-now').disabled,true);
  assert.equal($('edit-now').disabled,false);
  cp.edited=cp.target_views;renderCheckpoint();
  assert.equal($('repair-now').disabled,false);
  assert.equal($('edit-now').disabled,true);assert.equal($('edit-schedule').disabled,true);
  cp.rgb_renderer='cpu_splats';renderCheckpoint();
  assert.equal($('edit-now').disabled,true);assert.equal($('repair-now').disabled,true);
  assert.match($('edit-readiness').textContent,/Recapture.*Viser/);
  $('mode').value='baseline';renderCheckpoint();assert.equal($('repair-now').disabled,false);
  cp.rgb_renderer='viser';$('mode').value='edited';renderCheckpoint();
  cp.edited=0;cp.reconstruction_ready=false;cp.reconstruction_readiness='Need two distinct cameras';renderCheckpoint();
  assert.equal($('repair-now').disabled,true);assert.equal($('edit-now').disabled,false);
  assert.match($('repair-readiness').textContent,/two distinct cameras/);
  cp.reconstruction_ready=true;cp.complete=false;renderCheckpoint();
  assert.equal($('edit-now').disabled,true);
  assert.equal($('repair-now').disabled,true);
  // View finding and editing each schedule independently with immutable inputs.
  cp.complete=true;$('views').valid=true;$('views').value='10';
  schedule('select');const selection=pendingPayload;$('views').value='12';$('scene').value='changed';
  await queue('select','2099-01-02T00:00:00Z',selection);
  assert.equal(submitted.views,10);assert.equal(submitted.scene_id,'unrelated-current-scene');
  schedule('edit');const editing=pendingPayload;selected='/changed';
  await queue('edit','2099-01-03T00:00:00Z',editing);
  assert.equal(submitted.checkpoint,'/saved/cp');assert.equal(submitted.stage,'edit');
  state.jobs=[{run_id:'prior',preparation_supported:true,config:{splatfix:{source:'/original/dataset',model:'1.3b'}}}];
  $('benchmark-source').value='/unrelated/current/dataset';$('benchmark-model').value='14b';
  comparePreparation('prior');
  $('benchmark-model').value='1.3b';
  await Promise.resolve();
  assert.equal(submitted.source,'/original/dataset');
  assert.equal(submitted.model,'14b');
  assert.equal(submitted.preparation_from,'prior');
  assert.equal(submitted.resume_from,undefined);
  selected='/saved/cp';state.jobs=[];busy=false;cp.complete=true;cp.reconstruction_ready=true;
  $('reconstruction-method').value='oraclegs';$('oracle-capture-source').value='seva';
  renderCheckpoint();assert.equal($('edit-image-set').value,'seva');
  assert.equal($('repair-now').disabled,true);assert.equal($('seva-capture').disabled,false);
  assert.equal(payload('repair').oracle_capture_source,'seva');
  assert.equal(payload('edit').image_set,'seva');
  cp.seva_captured=cp.target_views;cp.seva_edited=0;renderCheckpoint();
  assert.equal($('seva-capture').disabled,true);assert.equal($('edit-now').disabled,false);
  assert.equal($('repair-now').disabled,true);
  $('mode').value='baseline';renderCheckpoint();assert.equal($('repair-now').disabled,false);
  cp.seva_edited=cp.target_views-1;renderCheckpoint();
  assert.equal($('edit-now').disabled,false);assert.match($('edit-readiness').textContent,/1 images need repair.*already saved and skipped/);
  $('mode').value='edited';cp.seva_edited=cp.target_views;renderCheckpoint();
  assert.equal($('edit-now').disabled,true);assert.equal($('edit-schedule').disabled,true);
  assert.equal($('repair-now').disabled,false);
  $('oracle-capture-source').value='saved';assert.equal(payload('repair').oracle_capture_source,'saved');
  // Switching away from a complete SEVA set enables capture of missing 77° RGBs.
  const oldTarget=cp.target_views;cp.target_views=cp.views.length;
  $('checkpoint-capture-set').value='77';renderCheckpoint();
  assert.equal(payload('seva_capture').capture_set,'77');
  assert.equal($('seva-capture').disabled,false);
  assert.match($('seva-capture-status').textContent,/77° captures/);
  state.jobs=[{config:{splatfix:{stage:'seva_capture',checkpoint:selected}},state:{status:'running'}}];
  renderCheckpoint();assert.equal($('seva-capture').disabled,true);
  state.jobs=[];
  for(const v of cp.views){v.fov77_original_url='/77-rgb.png';v.capture_77={rgb_renderer:'viser'};}
  $('edit-image-set').value='77';renderCheckpoint();
  assert.equal($('seva-capture').disabled,true);
  assert.equal($('edit-now').disabled,false);
  setCheckpointFov(cp.id,'77');
  assert.equal(checkpointImage(cp,cp.views[0]).url,'/77-rgb.png');
  for(const v of cp.views)v.fov77_repaired_url='/77-repaired.png';
  renderCheckpoint();assert.equal($('edit-now').disabled,true);
  assert.equal(checkpointImage(cp,cp.views[0]).url,'/77-repaired.png');
  $('checkpoint-capture-set').value='saved';
  assert.equal(payload('seva_capture').capture_set,'saved');
  setCheckpointFov(cp.id,'saved');

  cp.target_views=oldTarget;
  const view=cp.views[0];view.repaired_url='/saved-repair.png';view.seva_original_url='/seva-rgb.png';view.seva_repaired_url='/seva-repair.png';
  assert.equal(checkpointImage(cp,view).url,'/saved-repair.png');
  setCheckpointFov(cp.id,'seva');assert.equal(checkpointImage(cp,view).url,'/seva-repair.png');
  assert.match($('checkpoint-grid').innerHTML,/data-fov="seva"/);
  view.seva_repaired_url=null;assert.equal(checkpointImage(cp,view).url,'/seva-rgb.png');
  view.seva_original_url=null;renderCheckpointGallery();
  assert.equal(checkpointImage(cp,view).url,null);
  assert.match($('checkpoint-grid').innerHTML,/54° capture pending/);
  setCheckpointFov(cp.id,'saved');assert.equal(checkpointImage(cp,view).url,'/saved-repair.png');
  setCheckpointImageMode(cp.id,'rgb');assert.equal(checkpointImage(cp,view).url,view.original_url);
  assert.match($('checkpoint-grid').innerHTML,/checkpoint-toggle-divider/);
  assert.match($('checkpoint-grid').innerHTML,/data-image-mode="rgb"/);
  view.seva_original_url='/seva-rgb.png';view.seva_repaired_url='/seva-repair.png';
  setCheckpointFov(cp.id,'seva');assert.equal(checkpointImage(cp,view).url,'/seva-rgb.png');
  setCheckpointImageMode(cp.id,'gpt');assert.equal(checkpointImage(cp,view).url,'/seva-repair.png');
  view.seva_repaired_url=null;assert.equal(checkpointImage(cp,view).url,'/seva-rgb.png');
  assert.match(checkpointImage(cp,view).label,/RGB/);


})()`,context)).catch(error=>{console.error(error);process.exitCode=1});
'''
    result = subprocess.run([node, '-e', code, str(page)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_result_gallery_polling_stops_after_images_or_terminal_state():
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node is required for the dashboard JavaScript logic check')
    page = Path(__file__).resolve().parents[1] / 'src/splat_explorer/web/static/splatfix_result_detail.html'
    code = r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert/strict');
const source=fs.readFileSync(process.argv[1],'utf8').split('<script>')[1].split('</script>')[0];
const elements=new Map();
const get=id=>{if(!elements.has(id))elements.set(id,{value:'',textContent:'',innerHTML:'',addEventListener(){},querySelector(){return{}},querySelectorAll(){return[]},before(){},after(){},replaceWith(){},setAttribute(){},open:false});return elements.get(id)};
let calls=0,timer=null,lastDelay=null;
let result={run_id:'run',status:'running',kind:'ArtiFixer3D',metrics:{},gallery:{frames:[]}};
const document={getElementById:get,querySelectorAll:()=>[],hidden:false,addEventListener(){}};
const context=vm.createContext({document,window:{addEventListener(){}},location:{search:'?id=run'},URLSearchParams,
 ResizeObserver:class{observe(){}},
 fetch:async()=>{calls++;return{ok:true,json:async()=>result}},
 setTimeout:(fn,ms)=>{assert.ok([3000,5000,30000].includes(ms));lastDelay=ms;timer=fn;return 1},clearTimeout:()=>{timer=null}});
const flush=()=>new Promise(resolve=>setImmediate(resolve));
(async()=>{
 vm.runInContext(source,context);await flush();
 assert.equal(get('error').textContent,'');
 assert.equal(calls,1);assert.equal(typeof timer,'function');
 document.hidden=true;timer();await flush();assert.equal(calls,1);
 document.hidden=false;
 result={...result,inference_ready:true};timer();await flush();
 assert.equal(calls,2);assert.equal(timer,null);
 assert.match(get('refreshStatus').textContent,/Auto-refresh off/);
 // Manual refresh still retrieves the eventual reconstruction.
 await get('refresh').onclick();assert.equal(calls,3);assert.equal(timer,null);
 // Cancellation stays live even when the gallery is already complete.
 result={...result,status:'stopping'};
 await get('refresh').onclick();assert.equal(typeof timer,'function');assert.equal(lastDelay,5000);
 result={...result,status:'stopped'};timer();await flush();assert.equal(timer,null);

 for(const status of ['completed','error','stopped']){
   result={...result,status,inference_ready:false};await get('refresh').onclick();assert.equal(timer,null);
 }
 result={...result,status:'running',kind:'G4Splat · viewer approximation'};
 await get('refresh').onclick();assert.equal(typeof timer,'function');
 result={...result,g4splat:{input_staging:{status:'exporting'},source_depth:{enabled:true},gate:{},
   views:[],view_count:0,depth_scale:{},generation:[],artifacts:[],fitting:[]}};
 await get('refresh').onclick();assert.equal(lastDelay,3000);
 assert.equal(get('error').textContent,'');
 assert.match(get('g4Debug').innerHTML,/Local source-depth staging: exporting/);
 result={...result,status:'completed'};
 await get('refresh').onclick();assert.equal(timer,null);
 result={...result,kind:'ArtiFixer3D',gallery:{frames:[],verified_inputs:true,expected_count:0}};
 await get('refresh').onclick();assert.equal(timer,null);
 for(const mode of ['within_trajectory','global']){
   get('oracleInit').value='source_splat';get('oracleAttentionBatching').value=mode;
   context.fetch=async(url,options)=>{
     assert.equal(url,'/api/splatfix/results/reconstruct-artifixer');
     assert.deepEqual(JSON.parse(options.body),{id:'run',initialization:'source_splat',oracle_attention_batching:mode});
     return {ok:true,json:async()=>({ok:true,run:{run_id:'new-run'}})};
   };
   await get('oracleStart').onclick();
   assert.equal(context.location.href,'/splatfix/results/run?id=new-run');
   assert.equal(get('oracleStart').disabled,false);
 }

})().catch(e=>{console.error(e);process.exitCode=1});
'''
    result = subprocess.run([node, '-e', code, str(page)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_dashboard_new_defaults():
    import re
    page = Path(__file__).resolve().parents[1] / 'src/splat_explorer/web/static/splatfix.html'
    html = page.read_text()
    for element, expected in [('mode', 'edited'), ('oracle-reference-mode', 'scene_memory'), ('oracle-preset', 'spiral_expanding'), ('oracle-source-reference', 'nearest')]:
        select = re.search(r'<select id="'+element+r'"[^>]*>(.*?)</select>', html).group(1)
        assert re.search(r'<option value="([^"]+)"', select).group(1) == expected
    assert re.search(r'id="oracle-preset-scale"[^>]*value="0.25"', html)


def test_gaussian_limit_click_edit_and_validation():
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node required for dashboard controls')
    page = Path(__file__).resolve().parents[1] / 'src/splat_explorer/web/static/splatfix.html'
    code = r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert/strict');
const html=fs.readFileSync(process.argv[1],'utf8');
assert.match(html,/id="artifixer-max-gaussians"[^>]*value="1000000"/);
assert.ok(html.includes("body.max_gaussians=Number($('artifixer-max-gaussians').value)"));
const elements={};
for(const id of ['artifixer-max-gaussians','artifixer-gaussian-limit-display'])elements[id]={hidden:id==='artifixer-max-gaussians',value:'1000000',textContent:'1,000,000',focus(){this.focused=true},select(){},addEventListener(){},setCustomValidity(s){this.error=s},reportValidity(){}};
const ctx=vm.createContext({$:id=>elements[id]});
vm.runInContext(html.slice(html.indexOf('function editGaussianLimit()'),html.indexOf('function renderArtifixerPipeline()')),ctx);
ctx.editGaussianLimit();
const input=elements['artifixer-max-gaussians'],display=elements['artifixer-gaussian-limit-display'];
assert.equal(input.hidden,false);assert.equal(display.hidden,true);assert.equal(input.focused,true);
input.value='2500000';assert.equal(ctx.saveGaussianLimit(),true);assert.equal(display.textContent,'2,500,000');assert.equal(input.hidden,true);
for(const value of ['0','-1','1.5','','NaN','9007199254740992']){ctx.editGaussianLimit();input.value=value;assert.equal(ctx.saveGaussianLimit(),false);assert.equal(input.hidden,false);}
input.value='500000';assert.equal(ctx.saveGaussianLimit(),true);assert.equal(display.textContent,'500,000');
'''
    result = subprocess.run([node, '-e', code, str(page)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_restart_button_uses_saved_run_id_and_prevents_duplicate_clicks():
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node required for queue action test')
    page = Path(__file__).resolve().parents[1] / 'src/splat_explorer/web/static/splatfix.html'
    code = r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert/strict');
const html=fs.readFileSync(process.argv[1],'utf8');
assert.ok(html.includes('job.restart_supported?`<button data-restart='));
assert.ok(html.includes("()=>restartJob(el.dataset.restart)"));
let calls=[],messages=[],resolveRequest,refreshed=0;
const ctx=vm.createContext({renderJobs(){},toast:(...args)=>messages.push(args),refresh:async()=>refreshed++,
api:(path,body)=>{calls.push([path,body]);return new Promise(resolve=>resolveRequest=resolve)}});
vm.runInContext(html.slice(html.indexOf('const restartingJobs='),html.indexOf('async function cancel(id)')),ctx);
(async()=>{
 const first=ctx.restartJob('failed-run');await ctx.restartJob('failed-run');
 assert.equal(calls.length,1);assert.equal(calls[0][0],'/restart');
 assert.equal(JSON.stringify(calls[0][1]),JSON.stringify({run_id:'failed-run'}));
 resolveRequest({job:{run_id:'new-run'}});await first;
 assert.equal(refreshed,1);assert.equal(messages[0][0],'Restart queued: new-run');
 ctx.api=async()=>{throw Error('Saved checkpoint has been removed')};
 await ctx.restartJob('failed-run');assert.deepEqual(messages[1],['Saved checkpoint has been removed',true]);
})().catch(e=>{console.error(e);process.exitCode=1});
'''
    result = subprocess.run([node, '-e', code, str(page)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
