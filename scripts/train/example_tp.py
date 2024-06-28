from functools import partial
from typing import Sequence
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics import Metric, MetricCollection
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributed._tensor.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)
from torch.utils.data import DataLoader, Dataset

from composer.loss import soft_cross_entropy
from composer.models import ComposerClassifier, ComposerModel
from composer.trainer.trainer import Trainer
from composer.utils import dist


class RandomClassificationDataset(Dataset):
    """Classification dataset drawn from a normal distribution.

    Args:
        shape (Sequence[int]): shape of features (default: (1, 1, 1))
        size (int): number of samples (default: 100)
        num_classes (int): number of classes (default: 2)
    """

    def __init__(self, shape: Sequence[int] = (1, 1), size: int = 100, num_classes: int = 2):
        self.size = size
        self.shape = shape
        self.num_classes = num_classes
        self.x = None
        self.y = None

    def __len__(self):
        return self.size

    def __getitem__(self, index: int):
        # Note: lazily generate data so it runs after Composer seeds everything, giving the same
        # dataset across multiple calls when using the same seed.
        if self.x is None:
            self.x = torch.randn(self.size, *self.shape)
        if self.y is None:
            self.y = torch.randint(0, self.num_classes, size=(self.size,))
        return self.x[index], self.y[index]

    def get_metrics(self, is_train: bool = False) -> dict[str, Metric]:
        return MulticlassAccuracy(num_classes=self.num_classes, average='micro')

class SimpleModel(ComposerModel):
    """Small classification model.

    Args:
        num_features (int): number of input features (default: 1)
        num_classes (int): number of classes (default: 2)
    """

    def __init__(
        self,
        num_features: int = 1,
        num_classes: int = 2,
        num_hidden: int = 8,
        device: str = 'cpu',
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.fc1 = torch.nn.Linear(num_features, num_hidden, device=device, bias=bias)
        # self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(num_hidden, num_classes, device=device, bias=bias)
        # self.softmax = torch.nn.Softmax(dim=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.fc2(F.relu(self.fc1(x))))
    
    def loss(self, outputs: Tensor, batch: tuple[Any, Tensor], *args, **kwargs) -> Tensor:
        _, targets = batch
        return self._loss_fn(outputs, targets, *args, **kwargs)

    def update_metric(self, batch: Any, outputs: Any, metric: Metric) -> None:
        _, targets = batch
        metric.update(outputs, targets)

dist.initialize_dist('gpu')

model = SimpleModel()
dataset = RandomClassificationDataset(size=10)
dataloader = DataLoader(dataset, sampler=dist.get_sampler(dataset))
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# state_dict_type = 'sharded'
state_dict_type = 'full'


fsdp_config = {
    'data_parallel_shard_degree': 1,
    'state_dict_type': state_dict_type,
}

layer_plan = {
    'fc1': ColwiseParallel(),
    'fc2': RowwiseParallel(),
}

tp_config = {
    'tensor_parallel_degree': 4,
    'layer_plan': layer_plan,
}

trainer = Trainer(
    model=model,
    optimizers=optimizer,
    train_dataloader=dataloader,
    parallelism_config={
        # 'fsdp': {
        #     **fsdp_config,
        # },
        'tp': {
            **tp_config,
        },
    },
    progress_bar=False,
    log_to_console=True,
    max_duration='3ba',
    # save_folder='./checkpoints',
    # save_interval='1ba',
    # save_overwrite=True,
)
trainer.fit()

# state_dict = trainer.state.state_dict()
# if state_dict_type == 'sharded' or dist.get_global_rank() == 0:
#     print('\n\n[1, Saved]' + '*' * 50 + '\n')
#     print(state_dict['model']['module.2.weight'])

# model2 = SimpleModel()
# load_path = './checkpoints/ep0-ba3/' if state_dict_type == 'sharded' else './checkpoints/ep0-ba3-rank0.pt'
# trainer2 = Trainer(
#     model=model2,
#     optimizers=optimizer,
#     train_dataloader=dataloader,
#     parallelism_config={
#         'fsdp': {
#             **fsdp_config,
#         },
#         # 'tp': {
#         #     **tp_config,
#         # },
#     },
#     progress_bar=False,
#     log_to_console=True,
#     max_duration='3ba',
#     save_folder='./checkpoints',
#     save_interval='1ba',
#     save_overwrite=True,
#     load_path=load_path,
#     # load_weights_only=True,
# )

# print('\n\n[1.1, Random Init]' + '*' * 50 + '\n')
# print(trainer2.state.state_dict()['model']['module.2.weight'])

# from composer.utils import checkpoint
# checkpoint.load_checkpoint(path='./checkpoints/ep0-ba3/', state=trainer2.state, logger=trainer2.logger)

# state_dict = trainer2.state.state_dict()
# if state_dict_type == 'sharded' or dist.get_global_rank() == 0:
#     print('\n\n[3, Loaded]' + '*' * 50 + '\n')
#     print(state_dict['model']['module.2.weight'])

# trainer2.fit()