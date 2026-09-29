"""Exercise the loading path with tiny tensors instead of a 63 GB checkpoint."""
from types import ModuleType, SimpleNamespace
import sys

import pytest


@pytest.mark.parametrize("bad_checkpoint", [False, True])
def test_empty_loading_preserves_buffers_and_strict_checkpoint(monkeypatch, bad_checkpoint):
    torch = pytest.importorskip("torch")
    pytest.importorskip("accelerate")
    from splat_explorer.scene_runs_ext.artifixer_bridge import load_eval_pipe

    class Transformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2, 2))
            self.register_buffer("frequencies", torch.tensor([1., 2.]))

        @classmethod
        def from_config(cls, *args, **kwargs):
            model = cls()
            assert model.weight.is_meta
            assert not model.frequencies.is_meta
            return model

    original = Transformer.from_config

    def get_pipe(opts, device):
        model = Transformer.from_config("test")
        assert model.weight.dtype == torch.bfloat16
        assert not model.weight.is_meta
        assert model.frequencies.tolist() == [1., 2.]
        return SimpleNamespace(transformer=model)

    def load(model, opts):
        state = {"weight": torch.full((2, 2), 3.), "frequencies": torch.tensor([1., 2.])}
        if bad_checkpoint:
            del state["weight"]
        model.load_state_dict(state, strict=True)

    for name, attrs in {
        "diffusers": {"WanTransformer3DModel": Transformer},
        "model_eval": {},
        "model_eval.run_inference": {"get_eval_pipe": get_pipe},
        "model_eval.checkpoint_loading": {"load_transformer_checkpoint": load},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    if bad_checkpoint:
        with pytest.raises(RuntimeError, match="Missing key"):
            load_eval_pipe(SimpleNamespace(), torch.device("cpu"))
    else:
        pipe = load_eval_pipe(SimpleNamespace(), torch.device("cpu"))
        assert pipe.transformer.weight.tolist() == [[3., 3.], [3., 3.]]
        assert not pipe.transformer.training
        assert not pipe.transformer.weight.requires_grad
    assert Transformer.from_config == original
