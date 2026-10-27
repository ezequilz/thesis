"""Lossless layer-wise cache residency for the pinned 14B causal student.

Keep the upstream sampler, attention processors, BF16 tensors and context sizes.
Only cache placement changes: selected layers reside in CPU RAM between calls.
Synchronous copies deliberately avoid overlapping staging buffers on the A100.
"""
from contextlib import contextmanager
import inspect
import json
from pathlib import Path
from types import MethodType

GIB = 2**30
CACHE_ARGUMENTS = ('kv_cache', 'crossattn_cache', 'neighbor_crossattn_cache')


def plan_residency(layer_bytes, gpu_available, host_available, *, gpu_reserve=12 * GIB,
                   host_reserve=4 * GIB):
    """Reserve activations plus one staged layer, then fit exact caches on GPU/CPU."""
    if not layer_bytes or any(size <= 0 for size in layer_bytes):
        raise ValueError('Expected positive cache sizes for every transformer layer')
    budget = gpu_available - gpu_reserve
    total = sum(layer_bytes)
    if total <= budget:
        return set()
    budget -= max(layer_bytes)  # One offloaded layer is live on CUDA during its forward.
    offloaded, host = set(), 0
    for index, size in enumerate(layer_bytes):
        offloaded.add(index)
        host += size
        if total - host <= budget:
            break
    if total - host > budget or host + host_reserve > host_available:
        raise RuntimeError(
            '14B student caches do not fit available GPU/host memory with activation headroom: '
            f'cache={total / GIB:.2f} GiB, GPU available={gpu_available / GIB:.2f} GiB, '
            f'CPU cache={host / GIB:.2f} GiB, host available={host_available / GIB:.2f} GiB. '
            'Free other workloads or explicitly choose a smaller run resolution; '
            'context, references and precision will not be reduced automatically.')
    return offloaded


def choose_residency(sizes, gpu_available, host_available, *, preserved=0):
    """Fit reference-heavy calls without changing their tensors or attention.

    Six repaired plus six generated references need both memory pools. Prefer
    generous reserves, then use the smaller native-A100 budgets when needed.
    Even the tightest plan reserves 8 GiB plus a staged layer on CUDA and 2 GiB
    on the host; inputs/weights already allocated are excluded from availability.
    """
    for gpu_margin, host_margin in ((12, 4), (10, 3), (8, 2)):
        try:
            layers = plan_residency(sizes, gpu_available, host_available,
                                   gpu_reserve=gpu_margin * GIB + preserved,
                                   host_reserve=host_margin * GIB)
            return layers, gpu_margin, host_margin
        except RuntimeError:
            if gpu_margin == 8:
                raise


def available_host_bytes():
    """Respect both node availability and Slurm cgroup limits (v1 and v2)."""
    info = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    available = int(info['MemAvailable'].split()[0]) * 1024
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        _, controllers, relative = line.split(':', 2)
        if controllers and 'memory' not in controllers.split(','):
            continue
        root = Path('/sys/fs/cgroup') / ('memory' if controllers else '')
        current = root / relative.lstrip('/')
        # Containers can expose the cgroup as their mount root. Check ancestors
        # too: the Slurm allocation can impose a tighter limit than the step.
        while current == root or root in current.parents:
            try:
                if controllers:
                    limit = (current / 'memory.limit_in_bytes').read_text().strip()
                    used = int((current / 'memory.usage_in_bytes').read_text())
                    prefix = 'total_'
                else:
                    limit = (current / 'memory.max').read_text().strip()
                    used = int((current / 'memory.current').read_text())
                    prefix = ''
                stats = dict(row.split() for row in (current / 'memory.stat').read_text().splitlines())
                if limit != 'max':
                    reclaimable = reclaimable_file_bytes(stats, v1=bool(prefix))
                    available = min(available, max(0, int(limit) - used + reclaimable))
            except FileNotFoundError:
                pass
            if current == root:
                break
            current = current.parent
    return available


def reclaimable_file_bytes(stats, *, v1=False):
    """Loaded mmap pages remain charged as active file cache after unmapping.

    Count clean, unmapped file pages as reclaimable, including active pages.
    Exclude mapped/shared/dirty/writeback memory conservatively; never count
    anonymous tensors or pinned CPU caches as reclaimable.
    """
    names = (('total_cache', 'total_mapped_file', 'total_shmem', 'total_dirty', 'total_writeback')
             if v1 else ('file', 'file_mapped', 'shmem', 'file_dirty', 'file_writeback'))
    return max(0, int(stats.get(names[0], 0)) - sum(int(stats.get(key, 0)) for key in names[1:]))


@contextmanager
def staged_caches(caches, device):
    """Temporarily materialize mutable cache dictionaries, preserving CPU storage."""
    import torch
    originals = []
    try:
        for cache in caches:
            if cache is None:
                continue
            backing = {key: value for key, value in cache.items() if torch.is_tensor(value)}
            originals.append((cache, backing))
            for key, value in backing.items():
                cache[key] = value.to(device=device, copy=True)
        yield
    finally:
        for cache, backing in originals:
            for key, value in list(cache.items()):
                if not torch.is_tensor(value):
                    continue
                target = backing.get(key)
                if target is not None and target.shape == value.shape and target.dtype == value.dtype:
                    target.copy_(value)  # Blocking D2H; never let the next layer reuse pending storage.
                    cache[key] = target
                else:
                    cache[key] = value.to(device='cpu', copy=True)


class StudentMemory:
    def __init__(self, pipe, log_path):
        self.pipe = pipe
        self.log_path = Path(log_path)
        self.offloaded = set()
        self.reference_tokens = 0
        self.text_tokens = 0
        self.initializer = pipe._initialize_kv_cache
        self.original_denoise = pipe.denoise_to_latents
        self.original_forwards = []

    def record(self, phase, **extra):
        import torch
        device = self.pipe.vae.device
        free, total = torch.cuda.mem_get_info(device)
        record = {'phase': phase, 'inference_model': '14b', 'cache_policy': 'lossless_layer_offload',
                  'allocated_gib': torch.cuda.memory_allocated(device) / GIB,
                  'reserved_gib': torch.cuda.memory_reserved(device) / GIB,
                  'peak_allocated_gib': torch.cuda.max_memory_allocated(device) / GIB,
                  'peak_reserved_gib': torch.cuda.max_memory_reserved(device) / GIB,
                  'device_free_gib': free / GIB, 'device_total_gib': total / GIB, **extra}
        with self.log_path.open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        print('SPLATFIX_STUDENT_MEMORY: ' + json.dumps(record), flush=True)

    def initialize(self, batch_size, frame_seq_length, num_frames):
        import torch
        pipe = self.pipe
        pipe.clear_inference_caches()
        layers = list(pipe.transformer.blocks)
        itemsize = torch.empty((), dtype=pipe.vae.dtype).element_size()
        tokens = frame_seq_length * num_frames
        sizes = [batch_size * (tokens + self.reference_tokens + self.text_tokens) *
                 block.attn1.inner_dim * 2 * itemsize for block in layers]
        # The existing optional bridge-preservation adapter keeps extra immutable
        # boundary K/V outside the normal cache dictionaries. Budget those too.
        preserved = sum(len(getattr(getattr(block.attn1, 'processor', None), 'latent_indices', ())) *
                        frame_seq_length * batch_size * block.attn1.inner_dim * 2 * itemsize
                        for block in layers)
        free, total = torch.cuda.mem_get_info(pipe.vae.device)
        # empty_cache above makes free reflect reclaimable allocator memory.
        gpu_available = free - max(0, total - 80 * GIB)
        host_available = available_host_bytes()
        self.offloaded, gpu_margin, host_margin = choose_residency(
            sizes, gpu_available, host_available, preserved=preserved)
        pipe.kv_cache1 = []
        for index, block in enumerate(layers):
            device = 'cpu' if index in self.offloaded else pipe.vae.device
            shape = (batch_size, tokens, block.attn1.heads, block.attn1.inner_dim // block.attn1.heads)
            pipe.kv_cache1.append({
                'k': torch.zeros(shape, dtype=pipe.vae.dtype, device=device),
                'v': torch.zeros(shape, dtype=pipe.vae.dtype, device=device),
                'global_end_index': torch.zeros(1, dtype=torch.long, device=device),
                'local_end_index': torch.zeros(1, dtype=torch.long, device=device),
            })
        self.record('cache_allocation', cache_frames=num_frames, spatial_tokens=frame_seq_length,
                    reference_tokens=self.reference_tokens, cpu_layers=sorted(self.offloaded),
                    estimated_cache_gib=sum(sizes) / GIB,
                    estimated_host_cache_gib=sum(sizes[i] for i in self.offloaded) / GIB,
                    preserved_boundary_cache_gib=preserved / GIB,
                    gpu_activation_headroom_gib=gpu_margin, host_headroom_gib=host_margin,
                    host_available_gib=host_available / GIB)

    def install(self):
        import torch
        pipe = self.pipe
        if getattr(pipe.transformer, '_cp_world_size', 1) != 1 or pipe.frames_per_block != 7:
            raise ValueError('Student cache offload requires the pinned single-GPU causal pipeline')
        denoise_signature = inspect.signature(self.original_denoise)

        def denoise(_pipe, *args, **kwargs):
            bound = denoise_signature.bind(*args, **kwargs)
            neighbors = bound.arguments.get('neighbors')
            prompt = bound.arguments['prompt']
            if not torch.is_tensor(prompt):
                raise ValueError('Step 4 student requires the saved encoded caption')
            # Neighbor latents have VAE spatial scale 8 and patch size 2.
            ph, pw = pipe.transformer.patch_size[1:]
            spatial = pipe.vae.config.scale_factor_spatial
            self.reference_tokens = (0 if neighbors is None else neighbors.shape[1] *
                                     (neighbors.shape[-2] // spatial // ph) *
                                     (neighbors.shape[-1] // spatial // pw))
            self.text_tokens = prompt.shape[-2]
            torch.cuda.reset_peak_memory_stats(pipe.vae.device)
            try:
                result = self.original_denoise(*args, **kwargs)
            except Exception:
                self.record('denoise_failed')
                raise
            self.record('denoise_complete')
            return result

        pipe.denoise_to_latents = MethodType(denoise, pipe)
        pipe._initialize_kv_cache = self.initialize
        for index, block in enumerate(pipe.transformer.blocks):
            original = block.forward
            signature = inspect.signature(original)
            if not all(key in signature.parameters for key in CACHE_ARGUMENTS):
                raise ValueError('Unexpected upstream transformer cache interface')
            self.original_forwards.append((block, original))

            def forward(_block, *args, _index=index, _original=original, _signature=signature, **kwargs):
                if _index not in self.offloaded:
                    return _original(*args, **kwargs)
                if torch.is_grad_enabled():
                    raise ValueError('Student cache offload is inference-only')
                bound = _signature.bind(*args, **kwargs)
                caches = [bound.arguments.get(key) for key in CACHE_ARGUMENTS]
                with staged_caches(caches, pipe.vae.device):
                    return _original(*args, **kwargs)

            block.forward = MethodType(forward, block)
        self.record('loaded')
        return self


def install(pipe, log_path):
    return StudentMemory(pipe, log_path).install()
