"""Legacy startup-hook path; implementation belongs to ArtiFixer."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parents[1] / "artifixer/python_compat/sitecustomize.py"))
