import torch
from composer.core import Callback, State, Event
from composer.loggers import Logger
import nvtx


class NsysProfileCallback(Callback):
    """Start NVTX profiling after a certain number of batches."""
    def __init__(self, start_batch: int = 10, stop_batch: int = None):
        self.start_batch = start_batch
        self.stop_batch = stop_batch
        self.started = False

    def fit_start(self, state: State, logger):  # <-- required
        pass  # Optional: could log setup info here

    def run_event(self, event: Event, state: State, logger: Logger):  # <-- required
        current_batch = state.timestamp.batch.value
        x = 0
        if event == Event.BATCH_START and current_batch == self.start_batch:
            torch.cuda.nvtx.range_push("nsys_callback")
            print(f"[NSYS] >>> Started profiling at batch {current_batch}")
            logger.log_metrics({'nsys_started': current_batch})
            self.started = True

        if self.stop_batch is not None and event == Event.BATCH_END and current_batch == self.stop_batch:
            torch.cuda.nvtx.range_pop()
            print(f"[NSYS] <<< Stopped profiling at batch {current_batch}")
            logger.log_metrics({'nsys_stopped': current_batch})
            self.started = False