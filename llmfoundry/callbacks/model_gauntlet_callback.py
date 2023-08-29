
from composer.core import Callback

__all__ = ['ModelGauntlet']

class ModelGauntlet(Callback):
    """The ModelGauntlet callback has been renamed to EvalGauntlet.

    We've created this dummy class, in order to alert anyone who may have been
    importing ModelGauntlet.
    """

    def __init__(
        self,
        *args,  # pyright: ignore [reportMissingParameterType]
        **kwargs):  # pyright: ignore [reportMissingParameterType]
        raise ImportError(
            'ModelGauntlet class is deprecated, please use EvalGauntlet')
