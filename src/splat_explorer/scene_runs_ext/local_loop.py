"""Inner inspection loop for five complementary observations of one artifact."""
from __future__ import annotations
import copy
from types import SimpleNamespace
import numpy as np
from ..agent.actions import Action


class LocalRepairLoop:
    def __init__(self, policy, artifact, step, rig, depth, max_move, *, views=5, max_turns=30, rotation_degrees=90.):
        self.policy, self.artifact = policy, artifact
        self.steps = [step]
        self.positions = [rig.position.copy()]
        self.origin = rig.position.copy()
        valid = np.asarray(depth) if depth is not None else np.array([])
        valid = valid[np.isfinite(valid) & (valid > 0)]
        self.move_limit = max_move
        self.min_baseline = max(float(np.median(valid))*.1 if valid.size else max_move*.1, 1e-5)
        self.move_limit = max(self.move_limit, 1e-5)
        self.views, self.max_turns, self.turns = views, max_turns, 0
        self.rotation_limit = rotation_degrees
        self.note = ''
        self.saved_tools = getattr(policy, '_tools', None)
        self.saved_task = getattr(policy, '_task', None)
        if self.saved_tools is not None:
            policy._tools = [copy.deepcopy(t) for t in self.saved_tools
                             if t['function']['name'] in ('move','move_toward','rotate','view_depth')]
            for name, desc in [('capture_repair_view','Accept this view only if the SAME damaged object remains clearly visible with useful overlap.'),
                               ('cancel_local_repair','Abandon collection if the same object cannot be observed reliably.')]:
                policy._tools.append({'type':'function','function':{'name':name,'description':desc,
                    'parameters':{'type':'object','properties':{},'additionalProperties':False}}})
        if self.saved_task is not None:
            policy._task = SimpleNamespace(system_prompt=lambda *args: (
                f'Your task is to acquire {self.views} complementary views of ONE damaged object '
                'or sub-scene, to reconstruct it from repaired images. You are the photographer: '
                'move the CAMERA around the stationary object. Do not rotate or change the object. '
                'The first view has already been accepted. Plan the remaining views to expose '
                'different sides, occlusions and depth relationships while retaining recognizable '
                'overlap of the same physical region. Aim for substantially different viewing '
                'angles (roughly 15–30 degrees apart where space permits), not tiny nearby frames. '
                'ArtiFixer will generate the small local camera trajectories later; do not duplicate '
                'that work. Use normal move and move_toward navigation to walk sideways/around the '
                'object, and rotate generously to keep looking at it. In-place rotation alone does '
                'not create parallax. Several navigation actions before capturing are encouraged. '
                'Inspect each fresh RGB image; use view_depth when useful. Call capture_repair_view '
                'only when this is a meaningfully different perspective of the SAME target. '
                'Avoid nearly identical views, unrelated objects, and viewpoints where the target '
                'is hidden. No waypoint jumps. If you lose the target, navigate back and reacquire '
                'it; cancel only when useful coverage is inaccessible. These accepted images will '
                'be edited by GPT-image and treated as the intended corrected reconstruction views. '
                'Target defect: ' + str(artifact.args.get('description','')) + '. Initial image region: '
                + str(artifact.args.get('image_region','current view')) +
                '. That image region identifies the object initially; its screen position will change.'
            ))

    def context(self):
        return (f'OBJECT COVERAGE: {len(self.steps)}/{self.views} accepted views; '
                f'steps {self.steps}. Move at most {self.move_limit:.5g} scene units; '
                f'Collect broad object-relative parallax, not micro-steps. Minimum new baseline '
                f'{self.min_baseline:.5g}. Accepted offsets from first camera: '
                f'{[np.round(p-self.origin,3).tolist() for p in self.positions]}. {self.note}')

    def restore(self):
        if self.saved_tools is not None: self.policy._tools = self.saved_tools
        if self.saved_task is not None: self.policy._task = self.saved_task

    def handle(self, action, step, rig):
        """Return (safe action, state). Only a completed collection can repair."""
        self.turns += 1
        if action.name == 'cancel_local_repair' or self.turns > self.max_turns:
            self.restore()
            return Action('cancel_local_repair'), 'cancelled'
        if action.name == 'capture_repair_view':
            distance = min(np.linalg.norm(rig.position-p) for p in self.positions)
            if distance < self.min_baseline:
                self.note = 'View rejected: move farther around the same object before capturing; insufficient parallax baseline.'
                return action, 'collecting'
            self.steps.append(step); self.positions.append(rig.position.copy()); self.note = ''
            if len(self.steps) == self.views:
                self.restore()
                args = {**self.artifact.args, 'regenerate':'yes', 'repair_scope':'local',
                        'view_steps':self.steps[:-1]}
                return Action('report_artifact',args), 'ready'
            return action, 'collecting'
        if action.name in ('move','move_toward'):
            return action.clamped(self.move_limit, self.rotation_limit), 'collecting'
        if action.name == 'rotate':
            safe = action.clamped(self.move_limit, self.rotation_limit)
            if safe.args.get('pitch_degrees') is not None:
                safe.args['pitch_degrees'] = float(np.clip(safe.args['pitch_degrees'],rig.pitch_deg-self.rotation_limit,rig.pitch_deg+self.rotation_limit))
            return safe, 'collecting'
        if action.name == 'view_depth': return action, 'collecting'
        self.note = f'{action.name} is unavailable during local collection.'
        return Action('view_depth'), 'collecting'
