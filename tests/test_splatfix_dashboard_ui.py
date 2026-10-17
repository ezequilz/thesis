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
const get=id=>{if(!elements.has(id))elements.set(id,{value:'',textContent:'',innerHTML:'',disabled:false,valid:true,opened:false,
  selectedOptions:[{textContent:'Other scene'}],classList:{toggle(){}},style:{},addEventListener(){},querySelectorAll(){return[]},
  reportValidity(){return this.valid},showModal(){this.opened=true},close(){this.opened=false}});return elements.get(id)};
const context=vm.createContext({document:{getElementById:get},fetch:()=>new Promise(()=>{}),setInterval(){},setTimeout(){},assert});
vm.runInContext(source,context);
Promise.resolve(vm.runInContext(`(async()=>{
  state={scenes:[],jobs:[],checkpoints:[{id:'/saved/cp',name:'Saved camera checkpoint',scene:'Original scene',complete:true,
    target_views:7,edited:7,rgb_renderer:'viser',trajectory_caches:1,reconstruction_ready:true,views:[]}]};
  $('reconstruction-method').value='artifixer';
  selected='/saved/cp';$('scene').value='unrelated-current-scene';$('views').value='7';$('mode').value='edited';
  assert.equal(payload('select').views,7);
  assert.equal(payload('select').scene_id,'unrelated-current-scene');
  assert.equal(payload('repair').checkpoint,'/saved/cp');
  assert.equal(payload('repair').mode,'edited');
  assert.equal(payload('repair').image_cache_insertion,false);
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
  schedule('repair');
  assert.equal($('schedule-dialog').opened,true);
  assert.match($('schedule-context').textContent,/Saved camera checkpoint/);
  assert.match($('schedule-context').textContent,/GPT-image improved/);
  const saved=pendingPayload;
  $('reconstruction-method').value='other-method';
  selected='/different-checkpoint';$('mode').value='baseline';$('regularization-profile').value='artifixer';
  let submitted;
  api=async(path,body)=>{submitted=body;return {}};refresh=async()=>{};
  await queue('repair','2099-01-01T00:00:00Z',saved);
  assert.equal(submitted.checkpoint,'/saved/cp');
  assert.equal(submitted.mode,'edited');
  assert.equal(submitted.image_cache_insertion,true);
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
const get=id=>{if(!elements.has(id))elements.set(id,{addEventListener(){},querySelector(){return{}},setAttribute(){},open:false});return elements.get(id)};
let calls=0,timer=null,lastDelay=null;
let result={run_id:'run',status:'running',kind:'ArtiFixer3D',metrics:{},gallery:{frames:[]}};
const document={getElementById:get,querySelectorAll:()=>[],hidden:false};
const context=vm.createContext({document,location:{search:'?id=run'},URLSearchParams,
 ResizeObserver:class{observe(){}},
 fetch:async()=>{calls++;return{ok:true,json:async()=>result}},
 setTimeout:(fn,ms)=>{assert.ok([3000,30000].includes(ms));lastDelay=ms;timer=fn;return 1},clearTimeout:()=>{timer=null}});
const flush=()=>new Promise(resolve=>setImmediate(resolve));
(async()=>{
 vm.runInContext(source,context);await flush();
 assert.equal(calls,1);assert.equal(typeof timer,'function');
 document.hidden=true;timer();await flush();assert.equal(calls,1);
 document.hidden=false;
 result={...result,inference_ready:true};timer();await flush();
 assert.equal(calls,2);assert.equal(timer,null);
 assert.match(get('refreshStatus').textContent,/Auto-refresh off/);
 // Manual refresh still retrieves the eventual reconstruction.
 await get('refresh').onclick();assert.equal(calls,3);assert.equal(timer,null);
 for(const status of ['completed','error','stopped']){
   result={...result,status,inference_ready:false};await get('refresh').onclick();assert.equal(timer,null);
 }
 result={...result,status:'running',kind:'G4Splat · viewer approximation'};
 await get('refresh').onclick();assert.equal(typeof timer,'function');
 result={...result,g4splat:{input_staging:{status:'exporting'},source_depth:{enabled:true},gate:{},
   views:[],view_count:0,depth_scale:{},generation:[],artifacts:[],fitting:[]}};
 await get('refresh').onclick();assert.equal(lastDelay,3000);
 assert.match(get('g4Debug').innerHTML,/Local source-depth staging: exporting/);
 result={...result,status:'completed'};
 await get('refresh').onclick();assert.equal(timer,null);
 result={...result,kind:'ArtiFixer3D',gallery:{frames:[],verified_inputs:true,expected_count:0}};
 await get('refresh').onclick();assert.equal(timer,null);
})().catch(e=>{console.error(e);process.exitCode=1});
'''
    subprocess.run([node, '-e', code, str(page)], check=True, capture_output=True, text=True)
