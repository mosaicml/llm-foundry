from llmfoundry.utils.registry_utils import import_file
from llmfoundry.utils.config_utils import (
    EVAL_CONFIG_KEYS,
    EvalConfig,
    log_config,
    make_dataclass_and_log_config,
    process_init_device,
)
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)
from llmfoundry.registry import models
from composer.models.huggingface import HuggingFaceModel
from llmfoundry.metrics import (
    DEFAULT_CAUSAL_LM_EVAL_METRICS,
    DEFAULT_CAUSAL_LM_TRAIN_METRICS,
)
from torchmetrics import Metric
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from llmfoundry.utils.builders import (
    build_tokenizer,
)
from transformers import (
    PreTrainedTokenizerBase,
)
import os
from .eval import main

class CustomComposerHFCausalLM(HuggingFaceModel):
    model = None

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        use_train_metrics: bool = True,
        **kwargs
    ):

        train_metrics, eval_metrics = CustomComposerHFCausalLM.build_metrics(
            use_train_metrics=use_train_metrics,
        )

        # Set up config args for the model construction and base classes
        super().__init__(
            model=self.model,
            shift_labels=True,
            tokenizer=tokenizer,
            metrics=train_metrics,
            eval_metrics=eval_metrics,
        )

    @staticmethod
    def build_metrics(
        use_train_metrics: bool,
        additional_train_metrics: Optional[List[str]] = None,
        additional_eval_metrics: Optional[List[str]] = None,
    ) -> Tuple[List[Metric], List[Metric]]:
        """Builds the training and evaluation metrics for the model.

        Args:
            use_train_metrics (bool): Whether to use training metrics.
            additional_train_metrics (Optional[List[str]]): Additional training metrics to include.
            additional_eval_metrics (Optional[List[str]]): Additional evaluation metrics to include.

        Returns:
            Tuple[List[Metric], List[Metric]]: A tuple containing the list of training metrics and evaluation metrics.
        """
        from llmfoundry.utils.builders import build_metric

        train_metric_names = DEFAULT_CAUSAL_LM_TRAIN_METRICS + (
            additional_train_metrics or []
        )
        train_metrics = [
            build_metric(metric, {}) for metric in train_metric_names
        ] if use_train_metrics else []
        eval_metric_names = DEFAULT_CAUSAL_LM_EVAL_METRICS + (
            additional_eval_metrics or []
        )
        eval_metrics = [
            build_metric(metric, {}) for metric in eval_metric_names
        ]

        return train_metrics, eval_metrics


models.register('custom_hf_causal_lm', func=CustomComposerHFCausalLM)


def evaluate(model, yaml_path, scripts_dir, tokenizer_name=None):
    with open(yaml_path) as f:
        cfg = om.load(f)
    if 'fsdp_config' in cfg:
        cfg.pop('fsdp_config')
        print('\n'*4, 'Ignoring `fsdp_config` in the yaml file', '\n'*4)
    # 4 states: tokenizer or not, tokenizer in cfg or not
    # need the tokenizer name only
    if tokenizer_name:
        print('\n'*4, 'Using tokenizer name provided in evaluate function', '\n'*4)
    elif 'models' in cfg and len(cfg.models) and 'tokenizer' in cfg.models[0]:
        tokenizer_name = cfg.models[0].tokenizer.name
        print('\n'*4, 'Using tokenizer name from the first model in yaml config', '\n'*4)
    else:
        tokenizer_name = model.config.name_or_path
        print('\n'*4, 'Using tokenizer name from the name_or_path of the model', '\n'*4)

    cfg_override = om.create(f"""
models:
-            
  model_name: {model.config.name_or_path}                 
  model:
    name: custom_hf_causal_lm
  tokenizer:
    name: {tokenizer_name}
    kwargs:
      model_max_length: {cfg.max_seq_len}
    """)
    print('\n'*4, f'Overriding yaml config with the following:\n{om.to_yaml(cfg_override)}', '\n'*4)
    cfg = om.merge(cfg, cfg_override)

    CustomComposerHFCausalLM.model = model

    os.chdir(scripts_dir)
    main(cfg)
