"""Compatibility import for :mod:`splat_explorer.splatfix.artifixer.inference_preview`.

New code should import the backend module directly.
"""
if __name__ == '__main__':
    import runpy
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    runpy.run_module('splat_explorer.splatfix.artifixer.inference_preview', run_name='__main__')
else:
    import sys
    from .artifixer import inference_preview as _implementation
    sys.modules[__name__] = _implementation
