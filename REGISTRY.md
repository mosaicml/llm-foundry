# LLM Foundry Registry

Some components of LLM Foundry are registrable. This means that you can register options for these components, and then use them in your yaml config, without forking the library.

## How to register

There are a few ways to register a new component:

### Python entrypoints

You can specify registered components via a Python entrypoint if you are building your own package with registered components.

For example, the following would register the `WandBLogger` class, under the key `wandb`, in the `llm_foundry.loggers` registry:

<!--pytest.mark.skip-->
```yaml
[build-system]
requires = ["setuptools>=42", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "foundry_registry"
version = "0.1.0"
dependencies = [
    "mosaicml",
    "llm-foundry",
]

[project.entry-points."llm_foundry.loggers"]
my_logger = "foundry_registry.loggers:MyLogger"
```

### Direct call to register

You can also register a component directly in your code:

<!--pytest.mark.skip-->
```python
from composer.loggers import LoggerDestination
from llmfoundry.registry import loggers

class MyLogger(LoggerDestination):
    pass

loggers.register("my_logger", func=MyLogger)
```

### Decorators

You can also use decorators to register components directly from your code:

<!--pytest.mark.skip-->
```python
from composer.loggers import LoggerDestination
from llmfoundry.registry import loggers

@loggers.register("my_logger")
class MyLogger(LoggerDestination):
    pass
```

For both the direct call and decorator approaches, if using the LLM Foundry train/eval scripts, you will need to provide the `code_paths` argument, which is a list of files need to execute in order to register your components. For example, you may have a file called `foundry_imports.py` that contains the following:

<!--pytest.mark.skip-->
```python
from foundry_registry.loggers import MyLogger
from llmfoundry.registry import loggers

loggers.register("my_logger", func=MyLogger)
```

You would then provide `code_paths` to the train/eval scripts in your yaml config:

<!--pytest.mark.skip-->
```yaml
...
code_paths:
  - foundry_imports.py
...
```


## Discovering registrable components
Coming soon
