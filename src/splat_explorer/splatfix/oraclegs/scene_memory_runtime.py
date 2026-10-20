"""Isolated SEVA adapter: reserve thirteen sparse-pass slots for scene memory.

Output indexing and dense-pass inputs stay native. Reserved slots replace only
padding; they are never decoded as targets or exported as repaired ground truth.
"""
import functools
import json
from pathlib import Path

import numpy as np
from PIL import Image


def install_scene_memory(evaluation, demo, request):
    import torch
    if len(request['frames']) != 13 or request['reference_count'] != 19:
        raise ValueError('Scene memory requires 8 scaffold + 5 memory + 6 saved references')
    original_stats = demo.infer_prior_stats

    @functools.wraps(original_stats)
    def stats(T, num_input_frames, num_total_frames, version_dict):
        if num_input_frames != 6:
            raise ValueError('Scene memory requires exactly six saved references')
        # Infer anchor density with eight free slots, while retaining a native
        # 21-frame interpolation pass. Upstream must not count memory as GT.
        result = original_stats([14, 21], num_input_frames, num_total_frames, version_dict)
        version_dict['T'] = [27, 21]
        return result
    demo.infer_prior_stats = stats

    state = {'sparse':False}
    original_chunk = evaluation.chunk_input_and_test

    @functools.wraps(original_chunk)
    def chunk(T, input_c2ws, test_c2ws, input_ords, test_ords, options,
              task='img2img', chunk_strategy='gt', gt_input_inds=None):
        state['sparse'] = chunk_strategy == 'gt'
        if state['sparse']:
            if T != 27 or len(gt_input_inds) != 6:
                raise ValueError('Unexpected scene-memory sparse context layout')
            T -= 13
        return original_chunk(T, input_c2ws, test_c2ws, input_ords, test_ords,
                              options, task, chunk_strategy, gt_input_inds)
    evaluation.chunk_input_and_test = chunk

    cached = {}
    def tensors(height, width):
        if (height, width) not in cached:
            images, poses, intrinsics = [], [], []
            w, h = request['resolution']
            if width != w or height < h:
                raise ValueError('Scene memory resolution changed unexpectedly')
            top = (height-h)//2
            for frame in request['frames']:
                with Image.open(frame['file_path']) as image:
                    if image.size != (w, h):
                        raise ValueError('Scene memory RGB calibration mismatch')
                    padded = Image.new('RGB', (width, height))
                    padded.paste(image, (0, top))
                    images.append(np.asarray(padded, dtype=np.float32).transpose(2, 0, 1)/127.5-1)
                poses.append((np.asarray(frame['transform_matrix']) @ np.diag([1., -1., -1., 1.]))[:3])
                k = np.asarray(frame['K'], dtype=np.float32).copy()
                k[1, 2] += top
                k[0] /= width
                k[1] /= height
                intrinsics.append(k)
            cached[height, width] = tuple(torch.as_tensor(np.stack(x), dtype=torch.float32)
                                          for x in (images, poses, intrinsics))
        return cached[height, width]

    original_value = evaluation.get_value_dict
    report = {'reference_count':19, 'saved_reference_mode':request['saved_reference_mode'],
              'sparse_context_length':27, 'dense_memory_frames':0, 'chunks':[]}

    @functools.wraps(original_value)
    def value(curr_imgs, curr_input_frame_indices, curr_c2ws, curr_Ks,
              curr_input_camera_indices, all_c2ws, camera_scale):
        if state['sparse']:
            if len(curr_imgs) != 27:
                raise ValueError('Scene memory expected a 27-frame sparse batch')
            extra = tensors(*curr_imgs.shape[-2:])
            curr_imgs, curr_c2ws, curr_Ks = curr_imgs.clone(), curr_c2ws.clone(), curr_Ks.clone()
            for target, source in zip((curr_imgs, curr_c2ws, curr_Ks), extra):
                target[14:27] = source.to(device=target.device, dtype=target.dtype)
            curr_input_frame_indices = sorted(set(i for i in curr_input_frame_indices if i < 14) | set(range(14, 27)))
            if len(curr_input_frame_indices) != 19:
                raise ValueError('Scene memory lost a saved reference or conditioned a generated target')
            report['chunks'].append({'conditioning_slots':curr_input_frame_indices,
                                     'scaffold_slots':list(range(14, 22)), 'memory_slots':list(range(22, 27))})
            path = Path(request['report_path'])
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(report, indent=2))
            temporary.replace(path)
        return original_value(curr_imgs, curr_input_frame_indices, curr_c2ws, curr_Ks,
                              curr_input_camera_indices, all_c2ws, camera_scale)
    evaluation.get_value_dict = value
