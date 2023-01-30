# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""Implements a Mosaic BERT wrapper around a :class:`.ComposerTransformer`."""

from __future__ import annotations

from typing import Optional

import transformers
from composer.metrics.nlp import (BinaryF1Score, LanguageCrossEntropy,
                                  MaskedAccuracy)
from composer.models.huggingface import HuggingFaceModel
from torchmetrics import MeanSquaredError
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification.matthews_corrcoef import MatthewsCorrCoef
from torchmetrics.regression.spearman import SpearmanCorrCoef

from examples.bert.src.bert_layers import (BertForMaskedLM,
                                           BertForSequenceClassification)

all = ['create_mosaic_bert_mlm', 'create_mosaic_bert_classification']


def create_mosaic_bert_mlm(pretrained_model_name: str = 'bert-base-uncased',
                           model_config: Optional[dict] = None,
                           tokenizer_name: Optional[str] = None,
                           gradient_checkpointing: Optional[bool] = False,
                           pretrained_checkpoint: Optional[str] = None):
    """Mosaic BERT masked language model based on |:hugging_face:| Transformers.

    For more information, see
    `Transformers. <https://huggingface.co/transformers/>`_.

    This function creates a Mosaic BERT, which includes several throughput
    optimizations not available in |:hugging_face:| BERT as well as
    architecture changes based on ALiBi and Gated Linear Units.

    Args:
        pretrained_model_name (str): Name of the Hugging Face model to
            instantiate. This will determine the default model configuration.
            Default: ``bert-base-uncased``.
        model_config (dict): A dictionary of user-specified configurations to
            update/add to the default model configuration.
        tokenizer_name (str, optional): Tokenizer name used to preprocess the
            dataset and validate the models inputs.
        gradient_checkpointing (bool, optional): Use gradient checkpointing.
            Default: ``False``.
        pretrained_checkpoint (str, optional): The pretrained checkpoint to
            initialize the model weights. If provided, the state dictionary
            stored at `pretrained_checkpoint` will be loaded into the model
            after initialization. Default: ``None``.

    .. code-block::

        {
        "_name_or_path": "bert-base-uncased",
        "architectures": ["BertForMaskedLM"],
        "attention_probs_dropout_prob": 0.0,
        "classifier_dropout": null,
        "gradient_checkpointing": false,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "model_type": "bert",
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "pad_token_id": 0,
        "position_embedding_type": "absolute",
        "transformers_version": "4.16.0",
        "type_vocab_size": 2,
        "use_cache": true,
        "vocab_size": 30522
        }

    To create a Mosaic BERT model for Masked Language Model pretraining:

     .. testcode::

         from examples.bert.src.mosaic import create_mosaic_bert_mlm
         model = create_mosaic_bert_mlm()
    """
    if not model_config:
        model_config = {}

    # By default, turn off attention dropout in Mosaic BERT (otherwise, Flash Attention will be off by default)
    if 'attention_probs_dropout_prob' not in model_config:
        model_config['attention_probs_dropout_prob'] = 0.0

    if not pretrained_model_name:
        pretrained_model_name = 'bert-base-uncased'

    config = transformers.AutoConfig.from_pretrained(pretrained_model_name,
                                                     **model_config)
    config.return_dict = False
    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    if pretrained_checkpoint is not None:
        model = BertForMaskedLM.from_pretrained(
            pretrained_checkpoint=pretrained_checkpoint, config=config)
    else:
        model = BertForMaskedLM(config)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()  # type: ignore

    # setup the tokenizer
    if tokenizer_name:
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name)

    metrics = [
        LanguageCrossEntropy(ignore_index=-100,
                             vocab_size=model.config.vocab_size),
        MaskedAccuracy(ignore_index=-100)
    ]

    hf_model = HuggingFaceModel(model=model,
                                tokenizer=tokenizer,
                                use_logits=True,
                                metrics=metrics)

    # Padding for divisibility by 8
    # We have to do it again here because wrapping by HuggingFaceModel changes it
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)
    hf_model.model.resize_token_embeddings(config.vocab_size)

    return hf_model


def create_mosaic_bert_classification(
        num_labels: int,
        pretrained_model_name: str = 'bert-base-uncased',
        model_config: Optional[dict] = None,
        tokenizer_name: Optional[str] = None,
        gradient_checkpointing: Optional[bool] = False,
        pretrained_checkpoint: Optional[str] = None):
    """Mosaic BERT classification model based on |:hugging_face:| Transformers.

    For more information, see `Transformers. <https://huggingface.co/transformers/>`_.

    This function creates a Mosaic BERT, which includes several throughput
    optimizations not available in |:hugging_face:| BERT as well as
    architecture changes based on ALiBi and Gated Linear Units.

    Args:
        num_labels (int): The number of classes in the classification task.
        pretrained_model_name (str): Name of the Hugging Face model to
            instantiate. This will determine the default model configuration.
            Default: ``bert-base-uncased``.
        model_config (dict): A dictionary of user-specified configurations to
            update/add to the default model configuration.
        tokenizer_name (str, optional): Tokenizer name used to preprocess the
            dataset and validate the models inputs.
        gradient_checkpointing (bool, optional): Use gradient checkpointing.
            Default: ``False``.
        pretrained_checkpoint (str, optional): The pretrained checkpoint to
            initialize the model weights. If provided,
            the state dictionary stored at `pretrained_checkpoint` will be
            loaded into the model after initialization. Default: ``None``.

    .. code-block::
        {
            "_name_or_path": "bert-base-uncased",
            "architectures": [
            "BertForSequenceClassification
            ],
            "attention_probs_dropout_prob": 0.0,
            "classifier_dropout": null,
            "gradient_checkpointing": false,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "id2label": {
            "0": "LABEL_0",
            "1": "LABEL_1",
            "2": "LABEL_2"
            },
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "label2id": {
            "LABEL_0": 0,
            "LABEL_1": 1,
            "LABEL_2": 2
            },
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "model_type": "bert",
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "pad_token_id": 0,
            "position_embedding_type": "absolute",
            "transformers_version": "4.16.0",
            "type_vocab_size": 2,
            "use_cache": true,
            "vocab_size": 30522
        }

    To create a Mosaic BERT model for classification:

     .. testcode::
        from mosaic_bert import create_mosaic_bert_classification
        model = create_mosaic_bert_classification(num_labels=3) # if the task has three classes.

    Note:
        This function can be used to construct a BERT model for regression by
        setting ``num_labels == 1``. This will have two noteworthy effects.
        First, it will switch the training loss to :class:`~torch.nn.MSELoss`.
        Second, the returned :class:`.ComposerModel`'s train/validation metrics
        will be :class:`~torchmetrics.MeanSquaredError` and
        :class:`~torchmetrics.SpearmanCorrCoef`. For the classifcation case
        (when ``num_labels > 1``), the training loss is
        :class:`~torch.nn.CrossEntropyLoss`, and the train/validation
        metrics are :class:`~torchmetrics.Accuracy` and
        :class:`~torchmetrics.MatthewsCorrCoef`, as well as
        :class:`.BinaryF1Score` if ``num_labels == 2``.
    """
    if not model_config:
        model_config = {}

    # By default, turn off attention dropout in Mosaic BERT (otherwise, Flash Attention will be off by default)
    if 'attention_probs_dropout_prob' not in model_config:
        model_config['attention_probs_dropout_prob'] = 0.0

    model_config['num_labels'] = num_labels

    if not pretrained_model_name:
        pretrained_model_name = 'bert-base-uncased'

    config = transformers.AutoConfig.from_pretrained(pretrained_model_name,
                                                     **model_config)
    assert transformers.AutoModelForSequenceClassification.from_config is not None, 'AutoModelForSequenceClassification has from_config method'

    config.return_dict = False
    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    if pretrained_checkpoint is not None:
        model = BertForSequenceClassification.from_pretrained(
            pretrained_checkpoint=pretrained_checkpoint, config=config)
    else:
        model = BertForSequenceClassification(config)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()  # type: ignore

    # setup the tokenizer
    if tokenizer_name:
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name)

    if num_labels == 1:
        # Metrics for a regression model
        metrics = [MeanSquaredError(), SpearmanCorrCoef()]
    else:
        # Metrics for a classification model
        metrics = [
            Accuracy(),
            MatthewsCorrCoef(num_classes=model.config.num_labels)
        ]
        if num_labels == 2:
            metrics.append(BinaryF1Score())

    hf_model = HuggingFaceModel(model=model,
                                tokenizer=tokenizer,
                                use_logits=True,
                                metrics=metrics)

    # Padding for divisibility by 8
    # We have to do it again here because wrapping by HuggingFaceModel changes it
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)
    hf_model.model.resize_token_embeddings(config.vocab_size)

    return hf_model
