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
                'duplicates, occluded views and unrelated objects. The first selected tile '
                'will be the reconstruction anchor. Use the printed tile numbers, not step IDs. '
                'Target: ' + target)
        return (
            f'Move the camera around the same stationary object for {self.candidates} turns. '
            'Use move, move_toward and rotate with normal navigation distances and angles. '
            'Translate sideways or around the object to create parallax; rotate to keep it '
            'framed. Rotation alone does not create parallax. Each resulting view is recorded '
            'automatically. Do not report artifacts or capture views. No maps or waypoint jumps. '
            'After these moves you will receive a numbered overview and choose five views '
            'for reconstruction. Call exactly one movement tool per turn. Target: ' + target)

    def observe(self, observation, step, rig):
        """Record the result of the previous movement, not the pre-movement frame."""
        if self.pending_move:
            self.steps.append(step)
            self.frames.append(np.asarray(observation).copy())
            self.rigs.append(copy.deepcopy(rig))
            self.pending_move = False
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
            _draw_label(draw, (x + 8, y + 6), str(index + 1))
        return np.asarray(sheet)

    def context(self):
        return (f'INNER LOOP: {len(self.steps)}/{self.candidates} movement views recorded. '
                + ('Select five numbered tiles. ' if self.selecting else 'Keep the target framed. ')
                + self.note)

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
        if self.turns > max(self.max_turns, self.candidates):
            self.restore()
            return Action('cancel_local_repair'), 'cancelled'
        if action.name in ('move', 'move_toward', 'rotate'):
            self.pending_move = True
            return action.clamped(self.move_limit, self.rotation_limit), 'collecting'
        self.note = 'Only move, move_toward and rotate are available during movement collection.'
        return Action('local_noop'), 'collecting'
