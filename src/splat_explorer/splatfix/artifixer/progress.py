"""Shared 14B progress snapshots and dashboard-aware remote observation."""
import json
from pathlib import Path
import re
import shlex
import time

from ..checkpoint import atomic_json
from ..jobs import read_json

ACTIVE_INTERVAL = 60
BACKGROUND_INTERVAL = 300
ACTIVITY_TTL = 45


def large_model(options):
    return options.get('inference_model') in ('14b', '14b-teacher') or options.get('teacher_inference') is True


def mark_active(root, *, now=None):
    """Local browser heartbeat only; never contact LRZ from an HTTP request."""
    atomic_json(Path(root) / '.dashboard-active.json', {'at': time.time() if now is None else now})


def planned_count(packet, plus=False):
    plan = packet.get('plan') or {}
    if not plan.get('frame_count'):
        return None
    # These helpers are CPU-only and match the sampler's segmentation exactly,
    # including seeded capacity, reverse endpoints and no-split trajectories.
    from .official_worker import inference_jobs
    from .teacher_inference import planned_clips
    trajectory = {**plan['trajectory'], 'frames': [{}] * plan['frame_count']}
    cfg = plan.get('runtime', {})
    anchors = [a['frame_index'] for a in trajectory['anchors']]
    return len(list(planned_clips(inference_jobs(plan, trajectory, plus=plus),
        cfg.get('teacher_max_frames', 37), anchors,
        seeding=bool(cfg.get('initial_image_seeding') or cfg.get('image_cache_insertion')),
        layout=cfg.get('teacher_clip_layout', 'segment'))))


def stage(label, current=0, total=None, *, status='pending', detail='Waiting', **extra):
    percent = min(100., max(0., current / total * 100)) if total else None
    if status != 'completed' and percent is not None:
        percent = min(99.9, percent)
    return {'label': label, 'current': current, 'total': total,
            'percent': 100. if status == 'completed' else percent,
            'status': status, 'detail': detail, **extra}


def summarize(packet, *, observed_at):
    state = packet.get('state', {})
    phase, status = state.get('failed_phase', state.get('phase')), state.get('status')
    terminal = status in ('error', 'stopped')
    if terminal and phase == 'finished':
        phase = ('infer' if not packet.get('teachers', {}).get('inference', {}).get('complete') else
                 'distill' if not packet.get('reconstruction_complete') else 'plus')
    result = {'observed_at': observed_at, 'poll_seconds': ACTIVE_INTERVAL, 'stages': {}}
    for name, label in (('inference', '14B inference'), ('plus', 'Post-reconstruction inference')):
        data = packet.get('teachers', {}).get(name, {})
        log = packet.get('logs', {}).get(name, '')
        active = phase == ('infer' if name == 'inference' else 'plus')
        completed = bool(data.get('complete')) or (name == 'inference' and
            (phase in ('distill', 'plus') or packet.get('reconstruction_complete')))
        clips = len(data.get('clips', []))
        total = data.get('planned_clip_count')
        if total is None and (data or active) and packet.get('plan', {}).get('runtime', {}).get('inference_model') == '14b-teacher':
            total = planned_count(packet, plus=name == 'plus')
        steps = data.get('teacher_steps') or packet.get('plan', {}).get('runtime', {}).get('teacher_steps', 50)
        # A completed clip's 50/50 line is not the next clip's progress.
        tail = log.rsplit('ARTIFIXER_MEMORY ', 1)[-1]
        matches = [int(a) for a, b in re.findall(r'\|\s*(\d+)\s*/\s*(\d+)\s*\[', tail) if int(b) == steps]
        step = min(steps, matches[-1]) if matches else 0
        detail = ('Complete' if completed else 'Stopped' if terminal and active else
                  f'Clip {min(clips + 1, total) if total else clips + 1}' + (f' / {total}' if total else '') +
                  (f' · denoising {step} / {steps}' if matches else ' · preparing') if active else
                  'Waiting for reconstruction' if name == 'plus' else 'Waiting for input preparation')
        result['stages'][name] = stage(label, clips * steps + step, total * steps if total else None,
            status='completed' if completed else status if terminal and active else 'active' if active else 'pending',
            detail=detail, completed_clips=clips, total_clips=total,
            completed_frames=sum(len(c.get('frame_indices', [])) for c in data.get('clips', []) if c.get('output') == name),
            expected_frames=data.get('expected_frame_count'))
    training = packet.get('logs', {}).get('reconstruction', '')
    iterations = re.findall(r'Training progress:[^\r\n]*?\|\s*(\d+)\s*/\s*(\d+)\s*\[', training)
    current, total = map(int, iterations[-1]) if iterations else (0, 30000)
    recorded = packet.get('training', {})
    if type(recorded.get('iteration')) is int and type(recorded.get('total')) is int and recorded['total'] > 0:
        current, total = recorded['iteration'], recorded['total']
    complete = packet.get('reconstruction_complete', False)
    active = phase == 'distill'
    result['stages']['reconstruction'] = stage('3D reconstruction', total if complete else current, total,
        status='completed' if complete else status if terminal and active else 'active' if active else 'pending',
        detail='Complete' if complete else 'Finalizing reconstruction' if current >= total else
        f'Iteration {current:,} / {total:,}' if active or current else 'Waiting for inference')
    return result


class ProgressObserver:
    """The one manager owns the remote cadence; tabs only refresh local data."""
    def __init__(self, root, container, remote, *, clock=time.time, sleep=time.sleep):
        self.root, self.container, self.remote = Path(root), container, remote
        self.clock, self.sleep = clock, sleep
        self.marker = self.root / 'remote-observation.json'
        self.stop_seen = False

    def interval(self):
        at = read_json(self.root.parent / '.dashboard-active.json', {}).get('at')
        return ACTIVE_INTERVAL if at is not None and 0 <= self.clock() - at < ACTIVITY_TTL else BACKGROUND_INTERVAL

    def due(self):
        previous = read_json(self.marker, {}).get('attempted_at')
        return previous is None or self.clock() - previous >= self.interval()

    def wait(self, should_stop):
        while True:
            if should_stop() and not self.stop_seen:
                self.stop_seen = True
                break  # Explicit cancellation is a control action, not polling.
            if self.due():
                break
            self.sleep(1)
        atomic_json(self.marker, {'attempted_at': self.clock()})

    def poll(self, ssh):
        script = Path(__file__).with_name('progress_probe.py').read_text()
        command = 'python3 -c ' + shlex.quote(script) + ' ' + shlex.quote(self.remote) + ' ' + shlex.quote(self.container)
        packet = json.loads(ssh(command))
        now = self.clock()
        try:
            progress = summarize(packet, observed_at=now)
            atomic_json(self.root / 'progress.json', progress)
        except (ValueError, KeyError, TypeError):
            # Bad telemetry must not cancel an otherwise healthy reconstruction.
            pass
        directory = self.root / 'gpu'
        directory.mkdir(exist_ok=True)
        for name, body in packet.get('logs', {}).items():
            (directory / ('progress-' + name + '.log')).write_text(body)
        return packet
