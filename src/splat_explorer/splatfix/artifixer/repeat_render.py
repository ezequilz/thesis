"""Render a repeated benchmark's repaired checkpoint with pinned upstream code."""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True)
    parser.add_argument('--split', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, args.repo)
    from data_processing.render_3dgrut_colmap import render_3dgrut_colmap
    from data_processing.artifixer3d import require_render_outputs
    from PIL import Image
    body = json.loads(args.split.read_text())
    scene, entry = next(iter(body['test'].items()))
    def path(key):
        p = Path(entry[key])
        return p if p.is_absolute() else args.split.parent / p
    transforms = json.loads(path('transforms_path').read_text())
    output = render_3dgrut_colmap(checkpoint=path('reconstruction_checkpoint'), colmap_dir=path('image_root'),
        output_root=args.split.parent / 'repaired-input-renders', experiment_name='repeat_input',
        selected_indices=path('selected_indices_path'), trajectory_path=path('transforms_path'),
        trajectory_output_subdir='', downsample_factor=1)
    require_render_outputs(output, len(transforms['frames']), path('selected_indices_path'))
    for index, frame in enumerate(transforms['frames']):
        size = (frame.get('w', transforms['w']), frame.get('h', transforms['h']))
        for folder in ('renders', 'opacity'):
            with Image.open(output / folder / f'{index:05d}.png') as image:
                if image.size != size:
                    raise ValueError(f'Rendered {folder} frame {index} has size {image.size}, expected {size}')
    for key, relative in (('render_dir', 'renders'), ('opacity_dir', 'opacity'), ('selected_indices_path', 'selected_indices.json')):
        entry[key] = str((output / relative).relative_to(args.split.parent))
    args.split.write_text(json.dumps(body, indent=2) + '\n')
    print('Verified freshly rendered RGB/opacity at calibrated resolution for all camera poses', flush=True)


if __name__ == '__main__':
    main()
