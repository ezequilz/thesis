"""Inner inspection loop for five nearby observations of one artifact."""
from __future__ import annotations
import copy
from types import SimpleNamespace
import numpy as np
from ..agent.actions import Action


class LocalRepairLoop:
    def __init__(self, policy, artifact, step, rig, depth, max_move, *, views=5, max_turns=30, step_fraction=.025, rotation_degrees=5.):
        self.policy, self.artifact = policy, artifact
        self.steps = [step]
        self.positions = [rig.position.copy()]
        self.origin = rig.position.copy()
        valid = np.asarray(depth) if depth is not None else np.array([])
        valid = valid[np.isfinite(valid) & (valid > 0)]
        self.move_limit = min(max_move, float(np.median(valid))*step_fraction) if valid.size else max_move*.1
        self.move_limit = max(self.move_limit, 1e-5)
        self.views, self.max_turns, self.turns = views, max_turns, 0
        self.rotation_limit = rotation_degrees
        self.note = ''
        self.saved_tools = getattr(policy, '_tools', None)
        self.saved_task = getattr(policy, '_task', None)
        if self.saved_tools is not None:
            policy._tools = [copy.deepcopy(t) for t in self.saved_tools
                             if t['function']['name'] in ('move','rotate','view_depth')]
            for name, desc in [('capture_repair_view','Accept this view only if the SAME damaged object remains clearly visible with useful overlap.'),
                               ('cancel_local_repair','Abandon collection if the same object cannot be observed reliably.')]:
                policy._tools.append({'type':'function','function':{'name':name,'description':desc,
                    'parameters':{'type':'object','properties':{},'additionalProperties':False}}})
        if self.saved_task is not None:
            policy._task = SimpleNamespace(system_prompt=lambda *args: (
                f'Collect {self.views} adjacent translated views of ONE artifact for local reconstruction. '
                'Keep the same physical object visible, with substantial overlap. Use small sideways '
                'or forward/back movements and small rotations. Never use waypoints, jumps, or '
                'move_toward. After each useful translation, inspect the image, then '
                'capture_repair_view. Rotation alone adds no parallax. Do not explore other objects '
                'or report new artifacts. Cancel if you lose the object. Target: '
                + str(artifact.args.get('description','')) + '. Region: '
                + str(artifact.args.get('image_region','current view'))))

    def context(self):
        return (f'LOCAL REPAIR COLLECTION: {len(self.steps)}/{self.views} accepted views; '
                f'steps {self.steps}. Move at most {self.move_limit:.5g} scene units; '
                f'keep this object visible. {self.note}')

    def restore(self):
        if self.saved_tools is not None: self.policy._tools = self.saved_tools
        if self.saved_task is not None: self.policy._task = self.saved_task

    def handle(self, action, step, rig):
        """Return (safe action, state). Only a completed collection can repair."""
        self.turns += 1
        if action.name == 'cancel_local_repair' or self.turns > self.max_turns:
            self.restore()
            return Action('cancel_local_repair'), 'cancelled'
        if np.linalg.norm(rig.position-self.origin) > self.move_limit*5:
            self.note = 'Local radius exceeded; cancel and choose a closer artifact.'
            self.restore()
            return Action('cancel_local_repair'), 'cancelled'
        if action.name == 'capture_repair_view':
            distance = min(np.linalg.norm(rig.position-p) for p in self.positions)
            if distance < self.move_limit*.25:
                self.note = 'View rejected: translate before capturing; duplicate/rotation-only pose.'
                return action, 'collecting'
            self.steps.append(step); self.positions.append(rig.position.copy()); self.note = ''
            if len(self.steps) == self.views:
                self.restore()
                args = {**self.artifact.args, 'regenerate':'yes', 'repair_scope':'local',
                        'view_steps':self.steps[:-1]}
                return Action('report_artifact',args), 'ready'
            return action, 'collecting'
        if action.name == 'move':
            return action.clamped(self.move_limit, self.rotation_limit), 'collecting'
        if action.name == 'rotate':
            safe = action.clamped(self.move_limit, self.rotation_limit)
            if safe.args.get('pitch_degrees') is not None:
                safe.args['pitch_degrees'] = float(np.clip(safe.args['pitch_degrees'],rig.pitch_deg-self.rotation_limit,rig.pitch_deg+self.rotation_limit))
            return safe, 'collecting'
        if action.name == 'view_depth': return action, 'collecting'
        self.note = f'{action.name} is unavailable during local collection.'
        return Action('view_depth'), 'collecting'
