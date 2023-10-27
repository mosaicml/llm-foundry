# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from packaging import version


def is_flash_v2_installed():
    try:
        import flash_attn as flash_attn
    except:
        return False
    return version.parse(flash_attn.__version__) >= version.parse('2.0.0')


def is_flash_v1_installed():
    try:
        import flash_attn as flash_attn
    except:
        return False
    return version.parse(flash_attn.__version__) < version.parse('2.0.0')
