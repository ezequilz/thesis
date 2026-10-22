"""Exercise the GPU page's actual cancel handler without contacting LRZ."""
import shutil
import subprocess
from pathlib import Path

import pytest


def test_gpu_cancel_confirmation_and_request():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for the GPU JavaScript logic check")
    page = Path(__file__).resolve().parents[1] / "src/splat_explorer/web/static/gpu.html"
    code = r'''
const fs = require('fs'), vm = require('vm'), assert = require('assert/strict');
const page = fs.readFileSync(process.argv[1], 'utf8');
const start = page.indexOf('$("jobs").onclick =');
const end = page.indexOf('\nif (pageVisible()) startAuto();', start);
const jobs = {}, requests = [], prompts = [], notes = [];
let accepted = false, failure = false, renders = 0;
const context = vm.createContext({
  $: () => jobs, Date,
  window: {confirm: message => { prompts.push(message); return accepted; }},
  postJson: async (url, body) => {
    requests.push({url, body});
    if (failure) throw new Error('Cancellation failed');
    return {message: 'Cancelled successfully'};
  },
  setNote: (...args) => notes.push(args), render: () => renders++,
});
vm.runInContext(page.slice(start, end), context);
const button = {dataset: {cancel: '5866526', running: '1'}, disabled: false};
const click = () => jobs.onclick({target: {closest: selector => selector === '[data-cancel]' ? button : null}});
(async () => {
  await click();
  assert.equal(prompts.length, 1);
  assert.match(prompts[0], /5866526.*R/);
  assert.equal(requests.length, 0);
  assert.equal(button.disabled, false);
  accepted = true;
  await click();
  assert.equal(requests.length, 1);
  assert.equal(requests[0].url, '/api/repair/gpu/cancel');
  assert.equal(requests[0].body.job_id, '5866526');
  assert.equal(requests[0].body.confirm, true);
  assert.equal(renders, 1);
  assert.deepEqual(notes[0], ['info', 'Cancelled successfully']);
  await click(); // Ignore repeated clicks while the button is disabled.
  assert.equal(requests.length, 1);
  button.disabled = false;
  button.dataset.running = '0';
  await click();
  assert.equal(prompts.length, 2);
  assert.equal(requests.length, 2);
  assert.equal(requests[1].body.confirm, false);
  assert.equal(renders, 2);
  button.disabled = false;
  button.dataset.running = '1';
  failure = true;
  await click();
  assert.equal(button.disabled, false);
  assert.deepEqual(notes.at(-1), ['err', 'Cancellation failed']);
})().catch(error => { console.error(error); process.exitCode = 1; });
'''
    subprocess.run([node, "-e", code, str(page)], check=True, capture_output=True, text=True)
