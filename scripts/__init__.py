# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import scripts.train as train
import scripts.data_prep as data_prep
import scripts.eval as eval
__all__ = [
   'train','data_prep', 'eval' 
]

__version__ = '0.2.0'
