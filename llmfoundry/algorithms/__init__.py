from composer.algorithms import (Alibi, GatedLinearUnits, GradientClipping,
                                 LowPrecisionLayerNorm)

from llmfoundry.registry import algorithms

algorithms.register('gradient_clipping', func=GradientClipping)
algorithms.register('alibi', func=Alibi)
algorithms.register('gated_linear_units', func=GatedLinearUnits)
algorithms.register('low_precision_layernorm', func=LowPrecisionLayerNorm)