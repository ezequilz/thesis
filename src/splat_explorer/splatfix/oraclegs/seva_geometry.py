"""Experimental RGB reprojection guidance for SEVA; not native depth conditioning.

Camera-Z source depth stays in scene units. Reprojection uses the unnormalized
OpenCV cameras captured before SEVA's ray normalization. No upstream files change.
"""
import inspect
import json
from pathlib import Path

import numpy as np
from PIL import Image


def project_reference(rgb, inverse, valid, source_pose, source_k, target_pose, target_k, size):
    """Forward splat a reference with a target-Z buffer; leave disocclusions empty.

    A one-pixel footprint avoids rounding cracks. Only the closest sample at each
    target pixel survives. This cannot discover occluders absent from the source.
    """
    width, height = size
    y, x = np.nonzero(valid & np.isfinite(inverse) & (inverse > 0))
    guide = np.zeros((height, width, 3), np.float32)
    mask = np.zeros((height, width), bool)
    if not len(x):
        return guide, mask
    rays = np.stack([x, y, np.ones_like(x)], axis=1) @ np.linalg.inv(source_k).T
    xyz = rays / inverse[y, x, None]
    transform = np.linalg.inv(target_pose) @ source_pose
    xyz = xyz @ transform[:3, :3].T + transform[:3, 3]
    positive = np.isfinite(xyz).all(1) & (xyz[:, 2] > 1e-6)
    xyz, colors = xyz[positive], rgb[y[positive], x[positive]]
    uvz = xyz @ target_k.T
    uv = uvz[:, :2] / uvz[:, 2:]
    # Bound before integer conversion to avoid overflow for near-plane samples.
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < width-1) & (uv[:, 1] >= 0) & (uv[:, 1] < height-1)
    uv, xyz, colors = uv[inside], xyz[inside], colors[inside]
    pixel = np.floor(uv).astype(np.int64)
    indices = np.concatenate([(pixel[:, 1]+dy)*width+pixel[:, 0]+dx
                              for dx, dy in ((0,0),(1,0),(0,1),(1,1))])
    depths = np.tile(xyz[:, 2], 4)
    distances = np.concatenate([np.sum((pixel + [dx,dy] - uv)**2, axis=1)
                                for dx, dy in ((0,0),(1,0),(0,1),(1,1))])
    colors = np.tile(colors, (4, 1))
    order = np.lexsort((distances, depths, indices))
    indices = indices[order]
    first = np.concatenate(([True], indices[1:] != indices[:-1])) if len(indices) else np.array([], bool)
    chosen = order[first]
    guide.reshape(-1, 3)[indices[first]] = colors[chosen]
    mask.ravel()[indices[first]] = True
    # Suppress uncertain coverage edges before conservative latent pooling.
    padded = np.pad(mask, 1)
    mask = np.logical_and.reduce([padded[dy:dy+height, dx:dx+width]
                                 for dy in range(3) for dx in range(3)])
    return guide, mask


def blend_prediction(prediction, guide, mask, strength):
    """Identical affine constraint on both CFG branches; retain uncovered values."""
    return prediction * (1-strength*mask) + guide * (strength*mask)


class ReferenceGuidance:
    def __init__(self, inputs):
        root = Path(inputs)
        manifest = json.loads((root/'inputs.json').read_text())
        try:
            from .source_geometry import validate_assets
        except ImportError:
            from source_geometry import validate_assets
        validate_assets(root, manifest)
        frames = json.loads((root/'scene/transforms.json').read_text())['frames']
        self.references = []
        axes = np.diag([1., -1., -1., 1.])
        for view, frame in zip(manifest['views'], frames):
            name = Path(view['image']).stem
            with np.load(root/'source-depth'/(name+'.npz'), allow_pickle=False) as data:
                inv, valid = data['invdepth'].copy(), data['valid'].astype(bool)
            rgb = np.asarray(Image.open(root/view['image']).convert('RGB'), dtype=np.float32)/127.5-1
            if inv.shape != rgb.shape[:2] or valid.shape != inv.shape or not np.isfinite(inv).all():
                raise ValueError('SEVA source depth and RGB dimensions/values disagree')
            pose = np.asarray(frame['transform_matrix']) @ axes
            k = np.array([[frame['fl_x'],0,frame['cx']], [0,frame['fl_y'],frame['cy']], [0,0,1.]])
            self.references.append((rgb, inv, valid, pose, k))
        if not self.references:
            raise ValueError('SEVA source guidance requires reference depth assets')
        self.extent = max(np.linalg.norm(np.ptp([r[3][:3,3] for r in self.references], axis=0)), 1e-6)

    def target(self, pose, k, size, source_reference=None):
        # Choose one reference to avoid averaging inconsistent edited RGBs.
        def score(ref):
            rotation = np.clip((np.trace(ref[3][:3,:3].T @ pose[:3,:3])-1)/2, -1, 1)
            return np.linalg.norm(ref[3][:3,3]-pose[:3,3])/self.extent + np.arccos(rotation)/np.pi
        index = (min(range(len(self.references)), key=lambda i: score(self.references[i]))
                 if source_reference is None else source_reference)
        if not 0 <= index < len(self.references):
            raise ValueError('Invalid assigned SEVA source reference')
        rgb, inv, valid, source_pose, source_k = self.references[index]
        guide, mask = project_reference(rgb, inv, valid, source_pose, source_k, pose, k, size)
        return guide, mask, index


def install_geometry_guidance(evaluation, request, report_path):
    """Guide denoised latent estimates in both passes, leaving input frames intact."""
    import torch
    import torch.nn.functional as F
    strength = float(request['strength'])
    if not np.isfinite(strength) or not 0 < strength <= 1:
        raise ValueError('SEVA geometry guidance strength must be in (0, 1]')
    references = ReferenceGuidance(request['inputs'])
    original_value, original_sample = evaluation.get_value_dict, evaluation.do_sample
    value_signature, sample_signature = inspect.signature(original_value), inspect.signature(original_sample)
    report = {'version':1, 'method':'source-depth-rgb-reprojection-latent-guidance',
              'experimental':True, 'strength':strength, 'chunks':[],
              'limitations':'No learned depth input; source-only z-buffer cannot detect unseen occluders; edited RGBs may disagree with source geometry.'}
    report_path = Path(report_path)

    def value(*args, **kwargs):
        bound = value_signature.bind(*args, **kwargs)
        # Clone before upstream centering/scaling; these are OpenCV scene cameras.
        poses = bound.arguments['curr_c2ws'].detach().cpu().numpy().copy()
        ks = bound.arguments['curr_Ks'].detach().cpu().numpy().copy()
        result = original_value(*args, **kwargs)
        result['_source_geometry_cameras'] = (poses, ks)
        return result

    def sample(*args, **kwargs):
        bound = sample_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments['value_dict']
        poses, ks = values['_source_geometry_cameras']
        height, width = bound.arguments['H'], bound.arguments['W']
        conditioned = values['cond_frames_mask'].detach().cpu().numpy()
        guides, masks, records = [], [], []
        for i, (raw_pose, normalized_k) in enumerate(zip(poses, ks)):
            pose = np.eye(4)
            pose[:3] = raw_pose[:3]
            k = normalized_k.copy()
            k[0] *= width
            k[1] *= height
            if conditioned[i]:
                guide, mask, source = np.zeros((height,width,3),np.float32), np.zeros((height,width),bool), None
            else:
                assigned = values.get('_ordered_pair_sources')
                guide, mask, source = references.target(pose, k, (width,height),
                    source_reference=assigned[i] if assigned is not None else None)
            guides.append(guide)
            masks.append(mask)
            records.append({'slot':i, 'conditioned':bool(conditioned[i]), 'source_reference':source,
                            'projected_valid_fraction':float(mask.mean())})
        ae = bound.arguments['ae']
        device = values['cond_frames'].device
        images = torch.as_tensor(np.stack(guides), device=device).permute(0,3,1,2)
        pixel_masks = torch.as_tensor(np.stack(masks)[:,None], device=device, dtype=images.dtype)
        # Do not perturb the upstream random stream through the extra VAE encode.
        with torch.inference_mode(), torch.random.fork_rng(devices=[device.index if device.index is not None else torch.cuda.current_device()]), torch.autocast('cuda'):
            evaluation.load_model(ae)
            try:
                latent = ae.encode(images, bound.arguments['encoding_t'])
            finally:
                evaluation.unload_model(ae)
        mask = (F.adaptive_avg_pool2d(pixel_masks, latent.shape[-2:]) >= 1-1e-6).to(latent.dtype)
        mask = -F.max_pool2d(-mask, 3, stride=1, padding=1)
        mask = mask.to(device=latent.device)
        for i, record in enumerate(records):
            record['latent_valid_fraction'] = float(mask[i].mean().item())
        original_denoiser = bound.arguments['denoiser']

        class GuidedDenoiser:
            def __getattr__(self, name):
                return getattr(original_denoiser, name)

            def __call__(self, *a, **kw):
                prediction = original_denoiser(*a, **kw)
                if prediction.shape[0] not in (len(poses), 2*len(poses)) or prediction.shape[1:] != latent.shape[1:]:
                    raise ValueError('Unexpected SEVA denoiser layout for source guidance')
                copies = prediction.shape[0]//len(poses)
                return blend_prediction(prediction, latent.repeat(copies,1,1,1).to(prediction),
                                        mask.repeat(copies,1,1,1).to(prediction), strength)

        bound.arguments['denoiser'] = GuidedDenoiser()
        result = original_sample(*bound.args, **bound.kwargs)
        report['chunks'].append({'slots':records, 'completed':result is not None})
        report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = report_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(report_path)
        return result

    evaluation.get_value_dict = value
    evaluation.do_sample = sample
