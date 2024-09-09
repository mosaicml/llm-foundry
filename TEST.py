from torch.distributed.tensor.parallel import ColwiseParallel
from omegaconf import OmegaConf as om
from composer.utils import TPConfig


layer_plan = {'up_proj': ColwiseParallel}
tp_config = TPConfig(layer_plan)

om.to_yaml(tp_config)

