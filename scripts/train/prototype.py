from composer import Trainer
from llmfoundry.models import MPTForCausalLM, MPTConfig
from torch.distributed.fsdp import fully_shard

make_model = lambda: MPTForCausalLM(
    MPTConfig(
        d_model=64,
        n_heads=2,
        n_layers=2,
        max_seq_len=128,
    )
)

tiny_mpt_composer_fsdp2 = make_model()

model_fsdp_1 = Trainer(model=make_model(), parallelism_config={'fsdp': {}})

model_fsdp_2 = fully_shard(tiny_mpt_composer_fsdp2)
