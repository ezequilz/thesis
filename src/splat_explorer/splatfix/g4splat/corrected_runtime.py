"""Build a run-local overlay; never modify the shared pinned upstream checkout."""
import hashlib
import json
from pathlib import Path
import shutil

try:
    from .geometry import CORRECTION_VERSION
except ImportError:  # Executed by official_runner.py in the isolated environment.
    from geometry import CORRECTION_VERSION

UPSTREAM_SHA256 = '6f55f8ef686bb90052bd5d7948f7005ba2ebd0f0d16f8f1ee042fc1c6d450aa6'


def corrected_source(source):
    if hashlib.sha256(source.encode()).hexdigest() != UPSTREAM_SHA256:
        raise ValueError('Unexpected pinned MASt3R source; refusing calibration patch')
    start = source.index('        # Modify images and intrinsics so that the principal points')
    end = source.index('        # Modify intrinsics according to MASt3R resizing', start)
    source = source[:start] + '''        # SplatFix: apply the crop to both pixels and calibration. Keep the
        # full-resolution image and MASt3R pointmap on exactly the same rays.
        from splatfix_geometry import prepare_calibrated_image
        image_calibration = []
        dst_img_path = os.path.join(output_dir, 'images')
        os.makedirs(dst_img_path, exist_ok=True)
        for idx_img, img_path in enumerate(filelist):
            destination = os.path.join(dst_img_path, os.path.basename(img_path))
            pointmap_wh = tuple(imgs[idx_img]['img'].shape[-2:][::-1])
            intrinsics[idx_img], calibration = prepare_calibrated_image(
                img_path, destination, intrinsics[idx_img], pointmap_wh)
            image_calibration.append(calibration)
            filelist[idx_img] = destination
        imgs = load_images(filelist, size=image_size, verbose=not silent)

''' + source[end:]
    source = source.replace('        if fix_rotation and fix_translation:\n',
                            '        previous_cams2world = cams2world.numpy().copy()\n        if fix_rotation and fix_translation:\n', 1)
    anchor = "        print_debug('output extrinsics:')"
    source = source.replace(anchor, '''        from splatfix_geometry import reanchor_points
        for i_ in range(len(pts3d)):
            pts3d[i_] = reanchor_points(pts3d[i_], previous_cams2world[i_],
                                       cams2world[i_].numpy())

''' + anchor, 1)
    source = source.replace("        'focals': focals.numpy().tolist(),", """        'intrinsics': scene.intrinsics.cpu().numpy().tolist(),
        'image_calibration': image_calibration,
        'focals': focals.numpy().tolist(),""", 1)
    # A non-positive global similarity scale reverses camera depth.
    source = source.replace('            ofs_est2calib =', '''            if not np.isfinite(scale_est2calib) or scale_est2calib <= 0:
                raise ValueError('Invalid calibrated camera alignment scale')
            ofs_est2calib =''', 1)
    compile(source, 'corrected_run_mast3r.py', 'exec')
    return source


def prepare_runtime(repo, destination):
    repo, destination = Path(repo).resolve(), Path(destination)
    original = (repo / 'mast3r/run_mast3r.py').read_text()
    patched = corrected_source(original)  # Verify before creating anything.
    destination.mkdir(parents=True, exist_ok=False)
    for child in repo.iterdir():
        if child.name in ('.git', 'mast3r'):
            continue
        (destination / child.name).symlink_to(child, target_is_directory=child.is_dir())
    mast3r = destination / 'mast3r'
    mast3r.mkdir()
    for child in (repo / 'mast3r').iterdir():
        if child.name not in ('run_mast3r.py', '__pycache__', 'splatfix_geometry.py'):
            (mast3r / child.name).symlink_to(child, target_is_directory=child.is_dir())
    (mast3r / 'run_mast3r.py').write_text(patched)
    helper = Path(__file__).with_name('geometry.py')
    shutil.copyfile(helper, mast3r / 'splatfix_geometry.py')
    provenance = {'version': CORRECTION_VERSION, 'upstream_source_sha256': UPSTREAM_SHA256,
                  'patched_source_sha256': hashlib.sha256(patched.encode()).hexdigest(),
                  'geometry_helper_sha256': hashlib.sha256(helper.read_bytes()).hexdigest(),
                  'shared_checkout_modified': False,
                  'changes': ['calibrated pixel crop', 'point-map reanchoring when restoring poses',
                              'positive global scale', 'mandatory pre-alignment reprojection gate']}
    (destination.parent / 'geometry-correction.json').write_text(json.dumps(provenance, indent=2) + '\n')
    return destination
