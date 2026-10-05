"""Process-local annotation compatibility for the provisioned Python 3.10.

Loaded only through splatfix's worker PYTHONPATH. The pinned authors' source
imports typing.Self; its Python 3.11 meaning is provided by the standard
backport on 3.10. No model, inference, optimizer, or rendering code changes.
"""
import sys
import typing

if not hasattr(typing, 'Self'):
    from typing_extensions import Self
    typing.Self = Self
    print('SPLATFIX_RUNTIME_COMPAT: typing.Self = typing_extensions.Self '
          '(Python < 3.11 annotation backport; upstream source unchanged)',
          file=sys.stderr, flush=True)
