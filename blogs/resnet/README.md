# Mosaic ResNet 

The most efficient recipes for training ResNets on ImageNet.  Follow the steps below to reproduce our results.

For more information on how to use Composer, please see our [docs](http://docs.mosaicml.com).

The following recipes are provided:

   | Recipe | Training Time | Speed Up Methods | Target Accuracy Range | Optimal Training Duration |
   | --- | --- | --- | --- | --- |
   | [resnet50_mild.yaml](recipes/resnet50_mild.yaml) | Short | `BCELoss`, `BlurPool`, `FFCV Dataloader`, `FixRes`, `Label Smoothing`, `Progressive Resizing` | ≤ 78.1% | ≤ 60 epochs | 
   | [resnet50_medium.yaml](recipes/resnet50_medium.yaml) | Longer | `BCELoss`, `BlurPool`, `FFCV Dataloader`, `FixRes`, `Label Smoothing`, `Progressive Resizing`, `MixUp`, `SAM` | 78.1%-79.5% | 60-240 epochs |
   | [resnet50_hot.yaml](recipes/resnet50_hot.yaml) | Longest |`BCELoss`, `BlurPool`, `FixRes`, `Label Smoothing`, `Progressive Resizing`, `MixUp`, `SAM`, `RandAugment`, `Stochastic Depth`, `MosaicML ColOut` | > 79.5% | ≥ 240 epochs |

## Prequisites

* [MosaicML's Resnet50 Recipes Docker Image](https://hub.docker.com/r/mosaicml/pytorch_vision/tags)
   * Tag: `mosaicml/pytorch_vision:resnet50_recipes`
   * The image pre-configured with the following dependencies
      * Composer Version: 0.7.1
      * PyTorch Version: 1.11.0
      * CUDA Version: 11.3
      * Python Version: 1.9
      * Ubuntu Version: 20.04
* [Docker](https://www.docker.com/) or your container orchestration framework of choice
* [Imagenet Dataset](http://www.image-net.org/)
    
## Running a Recipe

1. Launch a Docker container using the `mosaicml/pytorch_vision:resnet50_recipes` Docker Image on your training system.
   
   ```
   docker pull mosaicml/pytorch_vision:resnet50_recipes
   docker run -it mosaicml/pytorch_vision:resnet50_recipes
   ``` 
   **Note:** The `mosaicml/resnet50_recipes` Docker image can also be used with your container orchestration framework of choice.

1. Download the ImageNet dataset from http://www.image-net.org/.

1. Create the dataset folder and extract training and validation images to the appropriate subfolders.
   The [following script](https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh) can be used to faciliate this process.
   Be sure to note the directory path of where you extracted the dataset.

   **Note:** This tutorial assumes that the dataset is installed to the `/tmp/ImageNet` path.

1. The `mild` and `medium` recipes require converting the ImageNet dataset to FFCV format.  *This conversion step is only required to be performed once, once converted files can be stashed away for reuse with subsequent runs.*

   1. Download the helper conversion script:
   
      ```
      wget -P /tmp https://raw.githubusercontent.com/mosaicml/composer/v0.7.1/scripts/ffcv/create_ffcv_datasets.py
      ```

   1. Convert the training and validation datasets.

      ```
      python /tmp/create_ffcv_datasets.py --dataset imagenet --split train --datadir /tmp/ImageNet/
      python /tmp/create_ffcv_datasets.py --dataset imagenet --split val --datadir /tmp/ImageNet/
      ```

      **Note:** The helper scripts output the FFCV formatted dataset files to `/tmp/imagenet_train.ffcv` and `/tmp/imagenet_val.ffcv` 
      for the training and validation data, respectively.

1. Pick the recipe you would like to train with and kick off the training run.

   ```
   composer -n {num_gpus} train.py -f recipes/{recipe_yaml} --max_duration {duration}
   ```

   Replace `{num_gpus}`, `{recipe_yaml}` and `{duration}` with the total number of GPU's, the recipe configuration, and total duration in epochs to train with, respectively.
   For example:
   
   ```
   composer -n 8 train.py -f recipes/resnet50_mild.yaml --max_duration 32ep
   ```

   The example above will train on 8 GPU's using the `mild` recipe for 32 epochs.

   **Note:** The `mild` and `medium` recipes assume the training and validation data is stored at the `/tmp/imagenet_train.ffcv` and `/tmp/imagenet_val.ffcv` paths while the `hot` recipe assumes the original ImageNet dataset is stored at the `/tmp/ImageNet` path.  The default paths can be overridden by default, please run `composer -n {num_gpus} train.py -f {recipe_yaml} --help` for more recipe specific configuration information.
