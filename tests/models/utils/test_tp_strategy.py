from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, PrepareModuleInput
from torch.distributed._tensor import Replicate, Shard

from llmfoundry.models.mpt.modeling_mpt import ComposerMPTCausalLM
from llmfoundry.utils.builders import build_tp_strategy


from icecream import install
install()

def test_tp_strategy():

    tp_config = {
        'strategy': 'ffn',
        }

    model_cfg = {
        'name': 'mpt_causal_lm',
        'd_model': 128,
        'n_heads': 4,
        'n_layers': 3,
        'expansion_ratio': 1,
        'max_seq_len': 16,
        'vocab_size': 50368,
        'attn_config': {
            'attn_impl': 'flash',
        },
    }

    _expected_layer_plan = {
        'ffn': PrepareModuleInput(
                input_layouts = Shard(0),
                desired_input_layouts = Replicate(),
                use_local_output = True,
            ),
        'ffn.down_proj': RowwiseParallel(
                input_layouts = Shard(-1),
                output_layouts = Shard(0),
            ),
        'ffn.up_proj': ColwiseParallel(
                input_layouts = Replicate(),
                output_layouts = Shard(-1),
            )
    }
    expected_layer_plan = {f'model.transformer.blocks.{layer_idx}.{name}': layer_plan for name, layer_plan in _expected_layer_plan.items() for layer_idx in range(model_cfg['n_layers'])}

    model = ComposerMPTCausalLM(**model_cfg)
    layer_plan = build_tp_strategy(tp_config['strategy'], model)

    # Compare expected and actual layer plan
    for (n1, lp1), (n2, lp2) in zip(sorted(expected_layer_plan.items()), sorted(layer_plan.items())):
        assert n1 == n2
        assert type(lp1) == type(lp2)
        if isinstance(lp1, PrepareModuleInput):
            assert lp1.input_layouts == lp2.input_layouts
            assert lp1.desired_input_layouts == lp2.desired_input_layouts
            assert lp1.use_local_output == lp2.use_local_output
        elif isinstance(lp1, ColwiseParallel) or isinstance(lp1, RowwiseParallel):
            assert lp1.input_layouts == lp2.input_layouts
            assert lp1.output_layouts == lp2.output_layouts
            assert lp1.use_local_output == lp2.use_local_output

if __name__ == '__main__':
    test_tp_strategy()
