from typing import Any
import abc

from composer.core import Callback

class CallbackWithConfig(Callback, abc.ABC):
    def __init__(self, config: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        del config, args, kwargs
        pass