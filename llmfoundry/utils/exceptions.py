"""Custom exceptions for the LLMFoundry."""

class MissingHuggingFaceURLSplitError(Exception):
    """Error thrown when a split is not found in a Hugging Face dataset used by the dataloader."""

    def __init__(self) -> None:
        message = 'When using a HuggingFace dataset from a URL, you must set the ' + \
                    '`split` key in the dataset config.'
        super().__init__(message)

class NotEnoughDatasetSamplesError(Exception):
    """Error thrown when there is not enough data to train a model."""
    def __init__(self, dataset_name: str, split: str, dataloader_batch_size: int, world_size: int, full_dataset_size: int, minimum_dataset_size: int):
        self.dataset_name = dataset_name
        self.split = split
        self.dataloader_batch_size = dataloader_batch_size
        self.world_size = world_size
        self.full_dataset_size = full_dataset_size
        self.minimum_dataset_size = minimum_dataset_size
        message = (f'Your dataset (name={dataset_name}, split={split}) '
                        +
                        f'has {full_dataset_size} samples, but your minimum batch size '
                        +
                        f'is {minimum_dataset_size} because you are running on {world_size} gpus and '
                        +
                        f'your per device batch size is {dataloader_batch_size}. Please increase the number '
                        +
                        f'of samples in your dataset to at least {minimum_dataset_size}.')
        super().__init__(message)


