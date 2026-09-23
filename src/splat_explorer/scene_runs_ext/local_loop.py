"""Move around one artifact, then select five views from a numbered contact sheet."""
from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

from ..agent.actions import Action
from ..rendering.annotate import _draw_label


class LocalRepairLoop:
    def __init__(self, policy, artifact, step, rig, depth, max_move, *,
                 views=5, max_turns=30, rotation_degrees=90., candidates=10):
        self.policy, self.artifact = policy, artifact
        self.steps = []
        self.frames = []
        self.rigs = []
        self.move_limit, self.rotation_limit = max_move, rotation_degrees
        self.candidates = candidates
        self.max_turns = max_turns
        self.turns = 0
        self.selection_attempts = 0
        self.pending_move = False
        self.pending_rotation = False
        self.needs_rotation = False
        self.previous_rig = copy.deepcopy(rig)
        self.origin = rig.position.copy()
        self.up = rig.up.copy()
        self.current_height = 0.
        # Scene units vary; reject negligible travel relative to the move limit.
        self.min_baseline = max(1e-6, float(max_move) * .01)
        self.selecting = False
        self.sheet = None
        self.anchor_rig = None
        self.anchor_frame = None
        self.note = ''
        self.saved_tools = getattr(policy, '_tools', None)
        self.saved_task = getattr(policy, '_task', None)
        if self.saved_tools is not None:
            policy._tools = [copy.deepcopy(t) for t in self.saved_tools
                             if t['function']['name'] in ('move', 'move_toward', 'rotate')]
        if hasattr(policy, '_task'):
            policy._task = SimpleNamespace(system_prompt=self.system_prompt,
                observation_label="Image 1 - RGB view from your current pose:", image_detail="auto")

    def system_prompt(self, *args):
        target = (str(self.artifact.args.get('description', '')) + '. Initial region: '
                  + str(self.artifact.args.get('image_region', 'current view')))
        if self.selecting:
            return (
                'The supplied image is a numbered tiled overview of your recent camera moves, '
                'NOT a single current camera view. Choose exactly FIVE distinct tile numbers '
                'using select_repair_views. Prefer complementary perspectives of the SAME '
                'target, useful parallax, clear visibility and overlapping content. Avoid '
                'duplicates, occluded views and unrelated objects. Include higher and lower '
                'vantages when visible and useful, not only a horizontal row. The first selected tile '
                'will be the reconstruction anchor. Use the printed tile numbers, not step IDs. '
                'Target: ' + target)
        return (
            f'Collect {self.candidates} translated, re-aimed views of the same stationary target. '
            'Each candidate requires TWO stages: translate, inspect the new RGB, then rotate '
            'to re-center the SAME target. A translation alone is not recorded. '
            'move left/right strafes relative to CURRENT yaw without turning the camera; '
            'forward/back also preserves heading, and up/down changes height. '
            'rotate yaw_degrees is RELATIVE: positive turns right, negative turns left. '
            'pitch_degrees is ABSOLUTE: positive looks up, negative down, 0 is level. '
            'For a centered target, strafe right then yaw LEFT; strafe left then yaw RIGHT; '
            'move up then pitch down. Use the observed target position to correct framing. '
            'Actively vary camera HEIGHT as well as lateral position: include at least one '
            'view above and one below the starting height when clearance allows. Use '
            'move(direction="up", distance=...) to raise the camera in 3D and '
            'move(direction="down", distance=...) to lower it along the scene vertical axis. '
            'These are real translations; changing pitch only tilts at the same height. '
            'After raising the camera, rotate toward a more downward absolute pitch; after '
            'lowering it, toward a more upward pitch. Inspect RGB to choose the actual angle. '
            'Try small height steps comparable to your lateral steps, avoid floor/ceiling '
            'obstacles, and combine elevated/lowered views with the lateral arc. '
            'Start with small lateral steps (roughly 10-20% of target distance) and turns '
            'around 5-15 degrees, adapting to the RGB and collision feedback. Sweep a shallow '
            'arc across both sides, keeping target scale and substantial image overlap. '
            'Avoid repeatedly walking forward toward the target or drifting sideways with '
            'fixed yaw. move_toward approaches a pixel surface; it does NOT aim the camera '
            'and preserves height; it is mainly for adjusting distance, not generating '
            'lateral or vertical baseline. '
            'Rotation alone creates no parallax; blocked moves and repeated positions do '
            'not count. Do not report artifacts or capture views. No maps or waypoint jumps. '
            'After these moves you will receive a numbered overview and choose five views '
            'for reconstruction. Call exactly one movement tool per turn. Target: ' + target)

    def observe(self, observation, step, rig):
        """Record the result of the previous movement, not the pre-movement frame."""
        self.current_height = float(np.dot(rig.position - self.origin, self.up))
        if self.pending_move:
            travelled = np.linalg.norm(rig.position - self.previous_rig.position)
            self.needs_rotation = travelled >= self.min_baseline
            self.note = (f'Translated {travelled:.3f} units. Now rotate to re-center the target.'
                         if self.needs_rotation else
                         'Translation was blocked or negligible. Try another travel direction.')
            self.pending_move = False
        if self.pending_rotation:
            angle = np.degrees(np.arccos(np.clip(np.dot(
                rig.view_direction(), self.previous_rig.view_direction()), -1., 1.)))
            distinct = all(np.linalg.norm(rig.position - r.position) >= self.min_baseline
                           for r in self.rigs)
            if angle >= .1 and distinct:
                self.steps.append(step)
                self.frames.append(np.asarray(observation).copy())
                self.rigs.append(copy.deepcopy(rig))
                self.needs_rotation = False
                self.note = 'Re-aimed view recorded. Translate to a new vantage next.'
            elif not distinct:
                self.needs_rotation = False
                self.note = 'Repeated camera position: translate to a different vantage.'
            else:
                self.note = 'No meaningful rotation occurred. Turn toward the target.'
            self.pending_rotation = False
        if len(self.steps) >= self.candidates and not self.selecting:
            self.selecting = True
            self.sheet = self._contact_sheet()
            if hasattr(self.policy, "_task"):
                self.policy._task.observation_label = "Numbered overview of recent movement views (select five tiles):"
                self.policy._task.image_detail = "high"
            if self.saved_tools is not None:
                self.policy._tools = [{'type': 'function', 'function': {
                    'name': 'select_repair_views',
                    'description': 'Select exactly five numbered tiles; first is the anchor.',
                    'parameters': {'type': 'object', 'properties': {
                        'views': {'type': 'array', 'items': {'type': 'integer', 'minimum': 1,
                                  'maximum': len(self.steps)}, 'minItems': 5, 'maxItems': 5,
                                  'uniqueItems': True}}, 'required': ['views'],
                                  'additionalProperties': False}}}]
        return self.sheet if self.selecting else observation

    def _contact_sheet(self):
        # Keep every source pixel; do not shrink the overview to VLM frame size.
        # Tile numbers sit inside each view, same dark backing as the bird's-eye title.
        height, width = self.frames[0].shape[:2]
        columns = min(3, len(self.frames))
        rows = math.ceil(len(self.frames) / columns)
        sheet = Image.new('RGB', (columns * width, rows * height))
        draw = ImageDraw.Draw(sheet)
        for index, frame in enumerate(self.frames):
            x = (index % columns) * width
            y = (index // columns) * height
            sheet.paste(Image.fromarray(frame), (x, y))
            # Size 18 matches the bird's-eye title and is unreadably small on a full view.
            label_size = max(18, min(height, width) // 8)
            _draw_label(draw, (x + 8, y + 6), str(index + 1), size=label_size)
        return np.asarray(sheet)

    def context(self):
        heights = [float(np.dot(r.position - self.origin, self.up)) for r in self.rigs]
        height_note = (
            f'Height relative to start: {self.current_height:+.3f} units; '
            f'recorded range [{min(heights, default=0.):+.3f}, '
            f'{max(heights, default=0.):+.3f}]. '
        )
        if self.selecting:
            height_note += 'Tile camera poses (use RGB to judge visibility): ' + '; '.join(
                f'{i + 1}: {r.state_description()}, height={heights[i]:+.3f}'
                for i, r in enumerate(self.rigs)
            ) + '. '
        if not self.selecting:
            if not any(h >= self.min_baseline for h in heights):
                height_note += 'Still need a higher vantage: use move up if clear. '
            if not any(h <= -self.min_baseline for h in heights):
                height_note += 'Still need a lower vantage: use move down if clear. '
        return (f'INNER LOOP: {len(self.steps)}/{self.candidates} movement views recorded. '
                + ('Select five numbered tiles. ' if self.selecting else
                   'NEXT: rotate to re-center the target. ' if self.needs_rotation else
                   'NEXT: translate sideways around the target. ')
                + height_note + self.note)

    def restore(self):
        if self.saved_tools is not None:
            self.policy._tools = self.saved_tools
        if hasattr(self.policy, '_task'):
            self.policy._task = self.saved_task

    def handle(self, action, step, rig):
        if self.selecting:
            self.selection_attempts += 1
            numbers = action.args.get('views', [])
            valid = (action.name == 'select_repair_views' and isinstance(numbers, list)
                     and len(numbers) == 5
                     and all(type(n) is int and 1 <= n <= len(self.steps) for n in numbers)
                     and len(set(numbers)) == 5)
            if valid:
                indices = [n - 1 for n in numbers]
                self.anchor_rig = self.rigs[indices[0]]
                self.anchor_frame = self.frames[indices[0]]
                self.selected_steps = [self.steps[i] for i in indices]
                self.restore()
                return Action('report_artifact', {**self.artifact.args, 'regenerate': 'yes',
                    'repair_scope': 'local', 'anchor_step': self.selected_steps[0],
                    'view_steps': self.selected_steps[1:]}), 'ready'
            self.note = 'Invalid selection. Choose exactly five distinct numbered tiles from the overview.'
            if self.selection_attempts >= 3:
                self.restore()
                return Action('cancel_local_repair'), 'cancelled'
            return Action('select_repair_views', action.args), 'collecting'
        self.turns += 1
        if self.turns > max(self.max_turns, 2 * self.candidates):
            self.restore()
            return Action('cancel_local_repair'), 'cancelled'
        if action.name in ('move', 'move_toward', 'rotate'):
            if self.needs_rotation and action.name != 'rotate':
                self.note = 'Rotate to re-center the target before translating again.'
                return Action('local_noop'), 'collecting'
            self.previous_rig = copy.deepcopy(rig)
            self.pending_move = action.name != 'rotate'
            self.pending_rotation = action.name == 'rotate' and self.needs_rotation
            self.note = ''
            return action.clamped(self.move_limit, self.rotation_limit), 'collecting'
        self.note = 'Only move, move_toward and rotate are available during movement collection.'
        return Action('local_noop'), 'collecting'
