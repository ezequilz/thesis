"""Checkpoint display names survive stage writes and do not change identity."""
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from splat_explorer.config import Config
from splat_explorer.scene_runs.store import SceneRunStore
from splat_explorer.splatfix.checkpoint import Checkpoint
from splat_explorer.web.splatfix_studio import SplatfixStudio


def test_display_name_persists_without_changing_checkpoint(tmp_path):
    studio = SplatfixStudio(SimpleNamespace(cfg=Config({'output': {'dir': str(tmp_path)}})),
                            SceneRunStore(tmp_path / 'scene-runs'))
    cp = Checkpoint.create(studio.checkpoint_root, '/fake/scene.ply', target_views=6)
    original = (cp.root / 'checkpoint.json').read_bytes()
    result = studio.rename_checkpoint({'checkpoint': str(cp.root), 'name': '  Kitchen / morning  '})
    assert result['name'] == 'Kitchen / morning'
    assert result['checkpoint'] == str(cp.root)
    assert (cp.root / 'checkpoint.json').read_bytes() == original
    cp.save()  # An already-loaded selection/edit worker can still save safely.
    entry = studio.checkpoints()[0]
    assert entry['name'] == 'Kitchen / morning'
    assert entry['storage_name'] == cp.root.name
    assert entry['id'] == str(cp.root)
    for name in ('', '  ', 'x' * 121, None, 123):
        with pytest.raises(ValueError):
            studio.rename_checkpoint({'checkpoint': str(cp.root), 'name': name})
    external = Checkpoint.create(tmp_path / 'outside', '/fake/scene.ply', target_views=6)
    with pytest.raises(ValueError):
        studio.rename_checkpoint({'checkpoint': str(external.root), 'name': 'Outside'})
    assert not (external.root / 'display-name.json').exists()


def test_gallery_selection_previews_and_rename():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node required')
    page = Path(__file__).resolve().parents[1] / 'src/splat_explorer/web/static/splatfix.html'
    script = r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert/strict');
const elements=new Map();
const get=id=>{if(!elements.has(id))elements.set(id,{value:'',innerHTML:'',style:{},handlers:{},
  classList:{toggle(){}},querySelectorAll(){return[]},addEventListener(type,fn){this.handlers[type]=fn},
  reportValidity(){return true},select(){},showModal(){this.open=true},close(){this.open=false}});return elements.get(id)};
const cropTimers=[];
const context=vm.createContext({document:{getElementById:get},fetch:()=>new Promise(()=>{}),setTimeout(fn,delay){if(delay===450)cropTimers.push(fn)},setInterval(){},assert,cropTimers});
vm.runInContext(fs.readFileSync(process.argv[1],'utf8').split('<script>')[1].split('</script>')[0],context);
vm.runInContext(`(async()=>{
  const views=Array.from({length:10},(_,i)=>({camera:{width:16,height:16},original_url:'/original'+i,repaired_url:i<6?'/repair'+i:null}));
  state.checkpoints=[{id:'/cp/a',name:'First',scene:'Room',views,target_views:10,edited:6},{id:'/cp/b',name:'Second',scene:'Room',views:[],target_views:6,edited:0}];
  selected='/cp/a';renderCheckpointOptions();renderCheckpoint();
  const gallery=$('checkpoint-grid').innerHTML;
  assert.equal((gallery.match(/<img /g)||[]).length,10);
  assert.ok(gallery.includes('src="/repair0"'));assert.ok(!gallery.includes('src="/original0"'));
  assert.ok(gallery.includes('src="/original9"'));assert.ok(gallery.includes('--columns:4'));
  assert.ok(gallery.includes('Waiting for saved views'));
  chooseCheckpoint('/cp/b');assert.equal($('checkpoint').value,'/cp/b');
  previewCheckpoint('/cp/a',5);
  assert.equal(selected,'/cp/b');assert.equal($('preview').open,true);
  assert.equal($('preview-image').src,'/repair5');assert.ok($('preview-caption').textContent.includes('View 6 of 10'));
  $('preview').handlers.keydown({key:'ArrowRight',preventDefault(){}});
  assert.equal($('preview-image').src,'/original6');
  $('preview').handlers.keydown({key:'ArrowLeft',preventDefault(){}});
  assert.equal($('preview-image').src,'/repair5');
  movePreview(1);assert.equal($('preview-image').src,'/original6');
  movePreview(20);assert.equal($('preview-image').src,'/original9');assert.equal($('preview-next').disabled,true);
  movePreview(1);assert.equal(previewIndex,9);
  movePreview(-20);assert.equal(previewIndex,0);assert.equal($('preview-previous').disabled,true);
  movePreview(-1);assert.equal(previewIndex,0);
  // Polling or changing the selected checkpoint must not alter the open sequence.
  const savedViews=state.checkpoints[0].views;state.checkpoints[0].views=[];
  movePreview(1);assert.equal($('preview-image').src,'/repair1');state.checkpoints[0].views=savedViews;
  assert.equal($('preview-display-controls').hidden,false);
  assert.ok($('preview-display-controls').innerHTML.includes('data-image-mode="rgb"'));
  assert.ok($('preview-display-controls').innerHTML.includes('data-fov="77"'));
  setPreviewDisplay('image','rgb');assert.equal($('preview-image').src,'/original1');
  assert.equal(previewIndex,1);assert.equal(selected,'/cp/b');assert.equal($('preview').open,true);
  setPreviewDisplay('fov','seva');assert.equal($('preview-image').hidden,true);
  assert.equal($('preview-missing').hidden,false);assert.equal(previewIndex,1);
  setPreviewDisplay('fov','77');assert.equal($('preview-image').hidden,true);
  setPreviewDisplay('fov','saved');assert.equal($('preview-image').src,'/original1');
  assert.equal($('preview-image').hidden,false);assert.equal($('preview-missing').hidden,true);
  setPreviewDisplay('image','gpt');assert.equal($('preview-image').src,'/repair1');
  assert.equal(checkpointImageModes.get('/cp/a'),'gpt');

  // Crop handles enforce 16:9 and minimum size in source-image pixels.
  const canvas=$('crop-canvas');canvas.getBoundingClientRect=()=>({left:0,top:0,width:800,height:450});
  const draws=[];
  canvas.getContext=()=>({drawImage(...args){draws.push(args)},fillRect(){},strokeRect(){},save(){},restore(){},beginPath(){},rect(){},clip(){}});canvas.setPointerCapture=()=>{};
  cropState={info:{width:1600,height:900},image:{kind:'gpt'},rgb:{kind:'rgb'},opacity:.65,box:[0,0,1600,900]};
  const pointer=(x,y)=>{const t=cropTransform();return {clientX:(t.frame[0]+(x-t.viewport[0])*t.scale)/2,clientY:(t.frame[1]+(y-t.viewport[1])*t.scale)/2,pointerId:1,preventDefault(){}}};
  drawCrop();const rgbBefore=draws.find(args=>args[0].kind==='rgb').slice(1);
  canvas.handlers.pointerdown(pointer(1600,900));
  canvas.handlers.pointermove(pointer(200,200));
  assert.deepEqual(cropState.box,[0,0,1280,720]);
  canvas.handlers.pointerup();assert.equal(cropTimers.length,1);
  assert.equal(cropState.viewport,undefined);cropTimers.shift()();
  assert.deepEqual(cropState.viewport,[0,0,1280,720]);
  // Widen the crop directly in the zoomed view, retaining full-source pixels.
  canvas.handlers.pointerdown(pointer(1280,720));
  canvas.handlers.pointermove(pointer(1440,810));
  assert.deepEqual(cropState.box,[0,0,1440,810]);
  canvas.handlers.pointerup();cropTimers.shift()();
  assert.deepEqual(cropState.viewport,[0,0,1440,810]);
  const rgbAfter=draws.filter(args=>args[0].kind==='rgb').at(-1).slice(1);
  assert.deepEqual(rgbAfter,rgbBefore);
  $('crop-opacity').value='0';$('crop-opacity').handlers.input();assert.equal(cropState.opacity,0);
  assert.deepEqual(cropState.box,[0,0,1440,810]);
  $('crop-opacity').value='100';$('crop-opacity').handlers.input();assert.equal(cropState.opacity,1);
  showFullCrop();assert.deepEqual(cropState.viewport,[0,0,1600,900]);
  canvas.handlers.pointerdown(pointer(720,405));
  canvas.handlers.pointermove(pointer(1800,1000));
  assert.deepEqual(cropState.box,[160,90,1440,810]);
  canvas.handlers.pointerup();stopCrop();cropTimers.shift()();assert.equal(cropState,null);
  $('preview').close();preview('/standalone');assert.equal($('preview-navigation').hidden,true);
  assert.equal($('preview-display-controls').hidden,true);
  movePreview(1);assert.equal($('preview-image').src,'/standalone');$('preview').close();

  assert.ok($('checkpoint-grid').innerHTML.includes('data-select-checkpoint="/cp/b" aria-pressed="true"'));
  $('checkpoint').value='/cp/a';chooseCheckpoint();assert.equal(selected,'/cp/a');
  renameCheckpoint('/cp/b');$('checkpoint-name').value='Kitchen <morning>';
  api=async(path,body)=>{assert.equal(path,'/checkpoints/rename');assert.equal(body.checkpoint,'/cp/b');return {checkpoint:body.checkpoint,name:body.name}};
  await $('rename-form').handlers.submit({preventDefault(){}});
  assert.equal(selected,'/cp/a');assert.equal(state.checkpoints[1].name,'Kitchen <morning>');
  assert.ok($('checkpoint').innerHTML.includes('>Kitchen &lt;morning&gt; · Room'));
  assert.equal($('rename-dialog').open,false);
  renameCheckpoint('/cp/b');api=async()=>{throw Error('Save failed')};
  await $('rename-form').handlers.submit({preventDefault(){}});
  assert.equal($('rename-dialog').open,true);assert.equal($('rename-error').textContent,'Save failed');
  $('rename-dialog').close();
  let deletes=0;
  api=async(path,body)=>{deletes++;assert.equal(path,'/checkpoints/delete');return {ok:true,checkpoint:body.checkpoint}};
  await requestDeleteCheckpoint('/cp/a');assert.equal(deletes,0);assert.equal($('delete-dialog').open,true);
  $('delete-dialog').close();assert.equal(state.checkpoints.length,2);
  await requestDeleteCheckpoint('/cp/b');assert.equal(deletes,1);assert.equal(selected,'/cp/a');assert.equal(state.checkpoints.length,1);
  await requestDeleteCheckpoint('/cp/a');
  api=async(path,body)=>{assert.equal(body.confirm_repaired,true);throw Error('In use')};
  await deleteCheckpoint(deleteId,true);assert.equal($('delete-error').textContent,'In use');assert.equal(state.checkpoints.length,1);
  api=async()=>({ok:true});await deleteCheckpoint(deleteId,true);
  assert.equal(selected,'');assert.equal(state.checkpoints.length,0);assert.equal($('checkpoint').value,'');
  assert.equal($('delete-dialog').open,false);assert.equal($('empty-checkpoints').hidden,false);
  state.checkpoints=[{id:'/new',name:'New',views:[],target_views:6,edited:0}];
  api=async()=>({requires_confirmation:true,checkpoint:'/new',name:'New',repaired:1});
  await requestDeleteCheckpoint('/new');assert.equal($('delete-dialog').open,true);assert.equal(state.checkpoints.length,1);

})()`,context).catch(error=>{console.error(error);process.exitCode=1});
'''
    subprocess.run([node, '-e', script, str(page)], check=True, capture_output=True, text=True)


def test_delete_checkpoint_confirmation_and_scope(tmp_path):
    import numpy as np
    from PIL import Image
    from splat_explorer.rendering.base import Camera
    studio = SplatfixStudio(SimpleNamespace(cfg=Config({'output': {'dir': str(tmp_path)}})),
                            SceneRunStore(tmp_path / 'scene-runs'))
    cp = Checkpoint.create(studio.checkpoint_root, '/fake/scene.ply', target_views=1)
    cp.add_view(Image.new('RGB', (16, 16)), Camera(np.zeros(3), np.eye(3), width=16, height=16, fov_deg=60))
    view = cp.views[0]
    view['repaired_rgb'] = view['original_rgb']
    cp.save()
    for confirmation in (None, False, 'true', 1):
        result = studio.delete_checkpoint({'checkpoint': str(cp.root), 'confirm_repaired': confirmation})
        assert result['requires_confirmation'] is True
        assert result['repaired'] == 1
        assert cp.root.exists()
    result = studio.delete_checkpoint({'checkpoint': str(cp.root), 'confirm_repaired': True})
    assert result['ok'] is True
    assert not cp.root.exists()
    plain = Checkpoint.create(studio.checkpoint_root, '/fake/scene.ply')
    assert studio.delete_checkpoint({'checkpoint': str(plain.root)})['ok']
    assert not plain.root.exists()
    external = Checkpoint.create(tmp_path / 'outside', '/fake/scene.ply')
    alias = studio.checkpoint_root / 'alias'
    alias.symlink_to(external.root, target_is_directory=True)
    for path in (tmp_path, studio.checkpoint_root, external.root, alias):
        with pytest.raises(ValueError):
            studio.delete_checkpoint({'checkpoint': str(path), 'confirm_repaired': True})
    assert external.root.exists()
    nested = Checkpoint.create(studio.root / 'run_test' / 'checkpoints', '/fake/scene.ply')
    assert studio.delete_checkpoint({'checkpoint': str(nested.root)})['ok']
    assert nested.root.parent.exists()


def test_delete_checkpoint_in_use(tmp_path):
    studio = SplatfixStudio(SimpleNamespace(cfg=Config({'output': {'dir': str(tmp_path)}})),
                            SceneRunStore(tmp_path / 'scene-runs'))
    cp = Checkpoint.create(studio.checkpoint_root, '/fake/scene.ply')
    run = SimpleNamespace(config=SimpleNamespace(pipeline='splatfix', splatfix={'checkpoint': str(cp.root)}),
                          state=SimpleNamespace(status=SimpleNamespace(value='queued'), details={}),
                          path=tmp_path / 'scene-runs' / 'run_test')
    studio.store.list_runs = lambda: [run]
    with pytest.raises(ValueError, match='active or queued'):
        studio.delete_checkpoint({'checkpoint': str(cp.root)})
    assert cp.root.exists()
    run.state.status.value = 'completed'
    assert studio.delete_checkpoint({'checkpoint': str(cp.root)})['ok']
