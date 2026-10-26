"""Shared low-host-memory loader for the pinned ArtiFixer pipelines."""
from unittest.mock import patch


def load_eval_pipe(opts, device):
    """Build the transformer without a full randomly initialized CPU copy.

    The release checkpoint replaces every parameter strictly. Keep real buffers
    (including rotary frequencies), but materialize empty BF16 parameters on the
    GPU before upstream adds its camera and neighbor-conditioning layers.
    """
    import torch
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from diffusers import WanTransformer3DModel
    from model_eval.run_inference import get_eval_pipe
    from model_eval.checkpoint_loading import load_transformer_checkpoint

    original = WanTransformer3DModel.from_config

    def empty_transformer(*args, **kwargs):
        with init_empty_weights(include_buffers=False):
            model = original(*args, **kwargs)
        for name, parameter in list(model.named_parameters()):
            value = torch.empty(parameter.shape, dtype=torch.bfloat16, device=device)
            set_module_tensor_to_device(model, name, device, value=value, dtype=torch.bfloat16)
        return model.to(device=device, dtype=torch.bfloat16)

    print("ARTIFIXER_LOADING constructing transformer directly on GPU (BF16)", flush=True)
    with patch.object(WanTransformer3DModel, "from_config", side_effect=empty_transformer):
        pipe = get_eval_pipe(opts, device)
    print("ARTIFIXER_LOADING copying memory-mapped checkpoint", flush=True)
    load_transformer_checkpoint(pipe.transformer, opts)
    pipe.transformer.eval().requires_grad_(False)
    print("ARTIFIXER_LOADED", flush=True)
    return pipe
