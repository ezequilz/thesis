"""Reference-grounded batching around the pinned authors' attention extractor."""
from pathlib import Path
import math
import shutil
import random


def plan_chunks(manifest, max_images=96, *, mode="within_trajectory", seed=23):
    """Shuffle target contexts, repeat every reference, and assign endpoint queries.

    Distance is trajectory parameter distance to its endpoints, not a global
    Euclidean pose metric. Midpoint ties deterministically use the start view.
    """
    if mode not in ("within_trajectory", "global"):
        raise ValueError("oracle_attention_batching must be within_trajectory or global")
    references = [Path(view['image']).name for view in manifest['views']]
    capacity = max_images-len(references)
    if not references or len(set(references)) != len(references) or capacity < 1:
        raise ValueError('Attention chunks require unique references and room for targets')
    segments = {s['index']:s for s in manifest['trajectory']['segments']}
    groups, seen = {}, set()
    for frame in manifest['trajectory']['frames']:
        name = 'synth_'+frame['image_name']
        if name in seen or Path(name).name != name:
            raise ValueError('Invalid or duplicate attention target')
        seen.add(name)
        alpha = frame['alpha']
        if not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError('Invalid attention trajectory position')
        segment = segments[frame['segment_index']]
        source = frame.get('reference_index')
        if source is None:
            source = segment['from_reference' if alpha <= .5 else 'to_reference']
        if not 0 <= source < len(references):
            raise ValueError('Attention source is not a saved reference')
        groups.setdefault(frame['segment_index'] if mode == 'within_trajectory' else None, []).append((frame,source))
    chunks = []
    rng = random.Random(seed)
    context_index = 0
    for group in groups.values():
        rng.shuffle(group)
        for start in range(0,len(group),capacity):
            context = group[start:start+capacity]
            context_names = ['synth_'+f['image_name'] for f,_ in context]
            for source in sorted({source for _,source in context}):
                selected = [f for f,i in context if i == source]
                ordered_refs = [references[source]]+[r for i,r in enumerate(references) if i != source]
                chunks.append({'index':len(chunks), 'context_index':context_index,
                    'segment_indices':sorted({f['segment_index'] for f,_ in context}),
                    'source_reference_index':source, 'source_image':references[source],
                    'source_frame':0, 'reference_images':ordered_refs,
                    'target_images':['synth_'+f['image_name'] for f in selected],
                    'context_target_images':context_names, 'image_names':ordered_refs+context_names,
                    'target_indices':[f['target_index'] for f in selected],
                    'alphas':[f['alpha'] for f in selected]})
            context_index += 1
    if not chunks:
        raise ValueError('Attention plan contains no targets')
    return chunks


def run_attention(request):
    # This module also runs as a sibling import in the isolated model worker.
    import numpy as np
    import torch
    import visualize_attn as upstream

    torch.manual_seed(request['seed'])
    root, destination = Path(request['image_dir']), Path(request['output_dir'])
    chunks = request['chunks']
    expected = [name for chunk in chunks for name in chunk['target_images']]
    actual = {p.name for p in root.glob('synth_*.png')}
    if len(expected) != len(set(expected)) or set(expected) != actual:
        raise ValueError('Attention plan does not cover every synthetic image exactly once')
    references = {p.name for p in root.glob('ref_*.png')}
    for chunk in chunks:
        names = chunk['image_names']
        if (names != chunk['reference_images']+chunk['context_target_images'] or
                names[0] != chunk['source_image'] or chunk['source_frame'] != 0 or
                len(names) > 96 or len(names) != len(set(names)) or
                set(chunk['reference_images']) != references or
                not set(chunk['target_images']).issubset(chunk['context_target_images'])):
            raise ValueError('Invalid reference-grounded attention chunk')
        if any(Path(name).name != name or not (root/name).is_file() for name in names):
            raise ValueError('Missing attention input image')
    destination.mkdir(parents=True,exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if device == 'cuda' and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model = upstream.VGGT()
    model.load_state_dict(torch.hub.load_state_dict_from_url(
        'https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt'))
    model.to(device).eval()
    for chunk in chunks:
        names = chunk['image_names']
        print(f"Attention chunk {chunk['index']+1}/{len(chunks)}: segments {chunk['segment_indices']}, "
              f"source {chunk['source_image']}, {len(chunk['reference_images'])} references, "
              f"{len(chunk['target_images'])} targets", flush=True)
        images = upstream.load_and_preprocess_images([str(root/name) for name in names], mode='pad').to(device).unsqueeze(0)
        B,S,_,H,W = images.shape
        patch_size = model.aggregator.patch_size
        patch_start = model.aggregator.patch_start_idx
        P = (H//patch_size)*(W//patch_size)+patch_start
        maps = []
        for layer in (0,22):
            with torch.no_grad(), torch.amp.autocast(device_type=device,dtype=dtype):
                predictions, raw = model(images,visualize_attn_maps=True,
                    vis_target_layer=layer,vis_source_frame=0)
                maps.append(upstream.normalize_attention_map(raw,P,patch_size,patch_start,H,W))
            del predictions, raw
        weighted = np.average(np.stack(maps),axis=0,weights=[.25,.75])
        # Keep reference diagnostics per chunk; publish each target exactly once.
        diagnostic = destination/f"chunk_{chunk['index']:03d}"
        upstream.save_attention_maps(weighted,B,S,patch_size,str(diagnostic),images,
            0,f"CHUNK_{chunk['index']:03d}",names,save_overlay=True)
        for name in chunk['target_images']:
            shutil.copyfile(diagnostic/name,destination/name)
        del images, maps, weighted
