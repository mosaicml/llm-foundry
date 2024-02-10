from composer import WandbLogger as ComposerWandbLogger

from llmfoundry import registry

@registry.loggers.register('wandb')
class WandBLogger(ComposerWandbLogger):
    pass