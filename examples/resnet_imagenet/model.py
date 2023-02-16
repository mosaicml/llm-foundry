# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import torch
from composer.loss import binary_cross_entropy_with_logits, soft_cross_entropy
from composer.metrics import CrossEntropy
from composer.models import ComposerClassifier
from torchmetrics import Accuracy, MetricCollection
from torchvision.models import resnet


def build_composer_resnet(model_name: str = 'resnet50',
                          loss_name: str = 'cross_entropy',
                          num_classes: int = 1000):
    """Helper function to build a Composer ResNet model.

    Args:
        model_name (str, optional): Name of the ResNet model to use, either
            ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']. Default: ``'resnet50'``.
        loss_name (str, optional): Name of the loss function to use, either ['cross_entropy', 'binary_cross_entropy'].
            Default: ``'cross_entropy'``.
        num_classes (int, optional): Number of classes in the classification task. Default: ``1000``.
    """
    model_fn = getattr(resnet, model_name)
    model = model_fn(num_classes=num_classes, groups=1, width_per_group=64)

    # Specify model initialization
    def weight_init(w: torch.nn.Module):
        if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(w.weight)
        if isinstance(w, torch.nn.BatchNorm2d):
            w.weight.data = torch.rand(w.weight.data.shape)
            w.bias.data = torch.zeros_like(w.bias.data)
        # When using binary cross entropy, set the classification layer bias to -log(num_classes)
        # to ensure the initial probabilities are approximately 1 / num_classes
        if loss_name == 'binary_cross_entropy' and isinstance(
                w, torch.nn.Linear):
            w.bias.data = torch.ones(
                w.bias.shape) * -torch.log(torch.tensor(w.bias.shape[0]))

    model.apply(weight_init)

    # Performance metrics to log other than training loss
    train_metrics = Accuracy()
    val_metrics = MetricCollection([CrossEntropy(), Accuracy()])

    # Choose loss function: either cross entropy or binary cross entropy
    if loss_name == 'cross_entropy':
        loss_fn = soft_cross_entropy
    elif loss_name == 'binary_cross_entropy':
        loss_fn = binary_cross_entropy_with_logits
    else:
        raise ValueError(
            f"loss_name='{loss_name}' but must be either ['cross_entropy', 'binary_cross_entropy']"
        )

    # Wrapper function to convert a image classification PyTorch model into a Composer model
    composer_model = ComposerClassifier(model,
                                        train_metrics=train_metrics,
                                        val_metrics=val_metrics,
                                        loss_fn=loss_fn)
    return composer_model
