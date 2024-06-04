# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from composer.callbacks import (
    EarlyStopper,
    Generate,
    LRMonitor,
    MemoryMonitor,
    MemorySnapshot,
    OOMObserver,
    OptimizerMonitor,
    RuntimeEstimator,
    SpeedMonitor,
)

from llmfoundry.callbacks.async_eval_callback import AsyncEval
from llmfoundry.callbacks.curriculum_learning_callback import CurriculumLearning
from llmfoundry.callbacks.eval_gauntlet_callback import EvalGauntlet
from llmfoundry.callbacks.eval_output_logging_callback import EvalOutputLogging
from llmfoundry.callbacks.fdiff_callback import FDiffMetrics
from llmfoundry.callbacks.hf_checkpointer import HuggingFaceCheckpointer
from llmfoundry.callbacks.log_mbmoe_tok_per_expert_callback import (
    MegaBlocksMoE_TokPerExpert,
)
from llmfoundry.callbacks.loss_perp_v_len_callback import \
    LossPerpVsContextLengthLogger
from llmfoundry.callbacks.monolithic_ckpt_callback import (
    MonolithicCheckpointSaver,
)
from llmfoundry.callbacks.resumption_callbacks import (
    GlobalLRScaling,
    LayerFreezing,
)
from llmfoundry.callbacks.run_timeout_callback import RunTimeoutCallback
from llmfoundry.callbacks.scheduled_gc_callback import ScheduledGarbageCollector
from llmfoundry.registry import callbacks, callbacks_with_config

callbacks.register('lr_monitor', func=LRMonitor)
callbacks.register('memory_monitor', func=MemoryMonitor)
callbacks.register('memory_snapshot', func=MemorySnapshot)
callbacks.register('speed_monitor', func=SpeedMonitor)
callbacks.register('runtime_estimator', func=RuntimeEstimator)
callbacks.register('optimizer_monitor', func=OptimizerMonitor)
callbacks.register('generate_callback', func=Generate)
callbacks.register('early_stopper', func=EarlyStopper)
callbacks.register('fdiff_metrics', func=FDiffMetrics)
callbacks.register('hf_checkpointer', func=HuggingFaceCheckpointer)
callbacks.register('global_lr_scaling', func=GlobalLRScaling)
callbacks.register('layer_freezing', func=LayerFreezing)
callbacks.register('mono_checkpoint_saver', func=MonolithicCheckpointSaver)
callbacks.register('scheduled_gc', func=ScheduledGarbageCollector)
callbacks.register('oom_observer', func=OOMObserver)
callbacks.register('eval_output_logging', func=EvalOutputLogging)
callbacks.register('mbmoe_tok_per_expert', func=MegaBlocksMoE_TokPerExpert)
callbacks.register('run_timeout', func=RunTimeoutCallback)

callbacks.register('loss_perp_v_len', func=LossPerpVsContextLengthLogger)

callbacks_with_config.register('async_eval', func=AsyncEval)
callbacks_with_config.register('curriculum_learning', func=CurriculumLearning)

__all__ = [
    'FDiffMetrics',
    'MonolithicCheckpointSaver',
    'GlobalLRScaling',
    'LayerFreezing',
    'ScheduledGarbageCollector',
    'EvalGauntlet',
    'HuggingFaceCheckpointer',
    'MegaBlocksMoE_TokPerExpert',
    'AsyncEval',
    'CurriculumLearning',
    'LossPerpVsContextLengthLogger',
]
