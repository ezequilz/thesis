import json
from types import SimpleNamespace
from splat_explorer.web.handoff_details import handoff_details


def test_live_download_then_repair_without_result_manifest(tmp_path):
    studio = SimpleNamespace(file_url=lambda p: str(p))
    (tmp_path / 'worker-request.json').write_text(json.dumps({'runtime': {'handoff_generation': True}}))
    folder = tmp_path / 'handoff' / ('a' * 32); folder.mkdir(parents=True)
    (folder / 'request.json').write_text(json.dumps({'id': folder.name, 'phase': 'inference', 'frame_index': 31}))
    (folder / 'before.png').write_bytes(b'png')
    result = handoff_details(studio, tmp_path)
    assert result['enabled']
    item = result['items'][0]
    assert item['frame_index'] == 31 and item['status'] == 'waiting for repair'
    assert list(item['images']) == ['before.png']
    (folder / 'repaired.png').write_bytes(b'fixed')
    (folder / 'response-original.png').write_bytes(b'raw')
    (folder / 'response.json').write_text(json.dumps({'id': folder.name, 'model': 'test'}))
    item = handoff_details(studio, tmp_path)['items'][0]
    assert item['status'] == 'repaired' and len(item['images']) == 3
    (folder / 'response.json').write_text(json.dumps({'error': 'Aspect mismatch'}))
    assert handoff_details(studio, tmp_path)['items'][0]['status'] == 'failed'


def test_archive_deduplicates_mailbox_and_rejects_external_symlink(tmp_path):
    root = tmp_path / 'run'; root.mkdir()
    studio = SimpleNamespace(file_url=lambda p: str(p))
    ident = 'b' * 32
    for folder in (root / 'handoff' / ident, root / 'gpu/results/edited/handoffs/inference/00031'):
        folder.mkdir(parents=True)
        (folder / 'response.json').write_text(json.dumps({'id': ident}))
        (folder / 'before.png').write_bytes(b'png')
    external = tmp_path / 'private.png'; external.write_bytes(b'private')
    (folder / 'repaired.png').symlink_to(external)
    items = handoff_details(studio, root)['items']
    assert len(items) == 1 and 'repaired.png' not in items[0]['images']


def test_handoff_ui_polling_and_partial_pairs(tmp_path):
    import shutil
    import subprocess
    from pathlib import Path
    import pytest
    node = shutil.which('node')
    if not node:
        pytest.skip('Node unavailable')
    page = Path(__file__).resolve().parents[1] / 'src/splat_explorer/web/static/splatfix_result_detail.html'
    script = r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert/strict');
const page=fs.readFileSync(process.argv[1],'utf8');
const elements=new Map();const $=id=>{if(!elements.has(id))elements.set(id,{innerHTML:'',value:'',hidden:false});return elements.get(id)};
let opened=null;const context=vm.createContext({$,esc:String,assert,choose:(...args)=>{opened=args}});
vm.runInContext(page.slice(page.indexOf('let savedArtifactGroup='),page.indexOf('let repeatStarting=')),context);
vm.runInContext(page.slice(page.indexOf('function needsGalleryPolling('),page.indexOf('function scheduleDetail(')),context);
vm.runInContext(`
const row={status:'running',kind:'ArtiFixer3D',inference_ready:true,handoff_debug:{enabled:true,items:[{id:'test',phase:'inference',frame_index:31,status:'waiting for repair',images:{before:{url:'/before.png',label:'Downloaded boundary'}}}]}};
assert.equal(needsGalleryPolling(row),true);
assert.equal(needsGalleryPolling({...row,status:'completed'}),false);
assert.equal(needsGalleryPolling({...row,handoff_debug:{enabled:false}}),true); // Reconstruction continues after inference.
showHandoffs(row);assert.match($('handoffDebug').innerHTML,/before.png/);assert.match($('handoffDebug').innerHTML,/waiting for repair/);
row.handoff_debug.items[0].images.repaired={url:'/repaired.png',label:'Repaired handoff'};
showHandoffs(row);assert.match($('handoffDebug').innerHTML,/repaired.png/);
assert.match($('handoffDebug').innerHTML,/data-handoff="1"/);
$('handoffDebug').onclick({target:{closest:()=>({dataset:{handoff:'1'}})}});
assert.equal(handoffFrames.length,2);assert.match(handoffFrames[1].label,/frame 31/);
const groups=[{name:'captures',images:[{url:'/capture.png',name:'capture'}]},{name:'generated',images:[{url:'/generated.png',name:'generated'}]}];
showSavedArtifacts({saved_artifacts:{image_groups:groups}});$('downloadGroup').value='1';$('downloadGroup').onchange();
assert.match($('downloadImages').innerHTML,/generated.png/);
showSavedArtifacts({saved_artifacts:{image_groups:groups}});assert.equal($('downloadGroup').value,'1');
`,context);
'''
    result = subprocess.run([node, '-e', script, str(page)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '<details id="savedArtifacts" hidden>' in page.read_text()
