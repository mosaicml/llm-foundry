from composer.loggers import MLFlowLogger as ComposerMLFlowLogger
from composer.utils import MissingConditionalImportError, dist
import json
import os
from composer.core.state import State
from composer.loggers.logger import Logger
from composer.loggers.logger_destination import LoggerDestination
from composer.utils import MissingConditionalImportError, dist



CONFIG_FILE = "/tmp/mlflow_config.yaml"
EXPERIMENT_ID_FIELD = "experiment_id"
RUN_ID_FIELD = "run_id"
TRACKING_URI_FIELD = "tracking_uri"


class MLFlowLogger(ComposerMLFlowLogger):
    
    def init(self, state: State, logger: Logger) -> None:
        super().init(state, logger)

        if self._enabled and dist.get_local_rank() == 0:
           if os.path.exists(CONFIG_FILE):
                os.remove(CONFIG_FILE)

           with open(CONFIG_FILE, "w") as f:
               data = {
                     EXPERIMENT_ID_FIELD: self._experiment_id,
                     RUN_ID_FIELD: self._run_id,
                     TRACKING_URI_FIELD : self.tracking_uri,
               }
               json.dump(data, f)