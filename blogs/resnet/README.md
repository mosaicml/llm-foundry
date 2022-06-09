<br />
<p align="center">
   <img src="https://assets.website-files.com/61fd4eb76a8d78bc0676b47d/62a185326fcd73061ab9aaf9_Hero%20Image%20Final.svg" width="50%" height="50%"/>
</p>

<h2><p align="center">The most efficient recipes for training ResNets on ImageNet</p></h2>

<h3><p align='center'>
<a href="https://www.mosaicml.com">[Website]</a>
- <a href="https://docs.mosaicml.com/">[Composer Docs]</a>
- <a href="https://docs.mosaicml.com/en/stable/method_cards/methods_overview.html">[Methods]</a>
- <a href="https://www.mosaicml.com/team">[We're Hiring!]</a>
</p></h3>

<p align="center">
    <a href="https://join.slack.com/t/mosaicml-community/shared_invite/zt-w0tiddn9-WGTlRpfjcO9J5jyrMub1dg">
        <img alt="Chat @ Slack" src="https://img.shields.io/badge/slack-chat-2eb67d.svg?logo=slack">
    </a>
    <a href="https://github.com/mosaicml/benchmarks/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-green.svg?logo=slack">
    </a>
</p>
<br />

# Prequisites

* [MosaicML's Resnet50 Recipes Docker Image](https://hub.docker.com/r/mosaicml/pytorch_vision/tags)
   * Tag: `mosaicml/pytorch_vision:resnet50_recipes`
   * The image comes pre-configured with the following dependencies:
      * Mosaic ResNet Training recipes
      * Training entrypoint: `train.py`
      * Composer Version: [0.7.1](https://github.com/mosaicml/composer/tree/v0.7.1)
      * PyTorch Version: 1.11.0
      * CUDA Version: 11.3
      * Python Version: 1.9
      * Ubuntu Version: 20.04
* [Docker](https://www.docker.com/) or your container orchestration framework of choice
* [Imagenet Dataset](http://www.image-net.org/)
* System with Nvidia GPUs
    
# Selecting a Recipe

As described in our [blog post](https://www.mosaicml.com/blog/mosaic-resnet):
> We actually cooked up three Mosaic ResNet recipes – which we call Mild, Medium, and Hot – to suit a range of requirements. 
> The Mild recipe is for shorter training runs, the Medium recipe is for longer training runs, and the Hot recipe is for the very 
> longest training runs that maximize accuracy. 

<img src="https://assets.website-files.com/61fd4eb76a8d78bc0676b47d/62a188a808b39301a7c3550f_Recipe%20Final.svg" width="50%" height="50%"/>

To reproduce a specific run, two pieces of information are required:

1. `recipe_yaml_path`: Path to the configuration file specifying the model and training parameters unique to each recipe.

1. `scale_schedule_ratio`: Factor which scales the duration of a particular run.

**Note:** The `scale_schedule_ratio` is a scaling factor for `max_duration`, each recipe sets a default `max_duration = 90ep`(epochs).  Thus a run with `scale_schedule_ratio = 0.1` will run for `90 * 0.3 = 27` epochs.

First, choose a recipe you would like to work with: [`Mild`, `Medium`, `Hot`].  This will determine which configuration file, `recipe_yaml_path`, you will need to specify. 

Next, determine the proper `scale_schedule_ratio` to specify to reproduce the desired run by using MosaicML's [Explorer](https://explorer.mosaicml.com).  Explorer enables users to identify the most cost effective way to run training workloads across clouds and on different types of hardware backends for a variety of models and datasets.  For this tutorial, we will focus on the [Mosaic ResNet run data](https://explorer.mosaicml.com/imagenet?sortBy=costSameQuality&model=resnet50&cloud=all&hardware=all&algorithms=all&baseline=r50_optimized_p4d&recipe=mosaicml_hot&recipe=mosaicml_medium&recipe=mosaicml_mild).

The table below provides the `recipe_yaml_path` for the selected recipe and a link to the corresponding Explorer page which can be used to select a specific run and obtain the corresponding value for `scale_schedule_ratio`:

   | Recipe | `recipe_yaml_path` | Explorer link |
   | --- | --- | --- |
   | Mild | `recipes/resnet50_mild.yaml` | [Mosaic Resnet Mild](https://explorer.mosaicml.com/imagenet?sortBy=timeSameQuality&model=resnet50&cloud=all&hardware=all&algorithms=all&baseline=r50_optimized_mosaicml&recipe=mosaicml_mild) |
   | Medium | `recipes/renset50_medium.yaml` | [Mosaic Resnet Medium](https://explorer.mosaicml.com/imagenet?sortBy=timeSameQuality&model=resnet50&cloud=all&hardware=all&algorithms=all&baseline=r50_optimized_mosaicml&recipe=mosaicml_medium) |
   | Hot | `recipes/resnet50_hot.yaml` | [Mosaic Resnet Hot](https://explorer.mosaicml.com/imagenet?sortBy=timeSameQuality&model=resnet50&cloud=all&hardware=all&algorithms=all&baseline=r50_optimized_mosaicml&recipe=mosaicml_hot) |

You can also compare all three recipes [here](https://explorer.mosaicml.com/imagenet?compare=recipe&sortBy=timeSameQuality&model=resnet50&cloud=mosaicml&hardware=all&algorithms=all&baseline=r50_optimized_mosaicml&recipe=mosaicml_hot&recipe=mosaicml_medium&recipe=mosaicml_mild&recipe=mosaicml_baseline).

In this tutorial we will using the `Mild` recipe and reproduce [this run](https://explorer.mosaicml.com/imagenet?sortBy=costSameQuality&selected=fks-short-timing-r6z2-seed-17-ssr0.32&model=resnet50&cloud=all&hardware=all&algorithms=all&baseline=r50_optimized_p4d&recipe=mosaicml_mild) which results in a Top-1 accuracy of 76.19%.  Thus, we see from the table above that the `recipe_yaml_path = recipes/resnet50_mild.yaml` and from Explorer that `scale_schedule_ratio = 0.32` for the desired run.

# Running a Recipe

Now that we've selected a recipe and determined the `recipe_yaml_path` and `scale_schedule_ratio` to specify, let's kick off a training run.

1. Launch a Docker container using the `mosaicml/pytorch_vision:resnet50_recipes` image on your training system.
   
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

1. The `Mild` and `Medium` recipes require converting the ImageNet dataset to FFCV format.  *This conversion step is only required to be performed once, once converted files can be stashed away for reuse with subsequent runs.*  The `Hot` recipe uses the original ImageNet data.

   1. Download the helper conversion script:
   
      ```
      wget -P /tmp https://raw.githubusercontent.com/mosaicml/composer/v0.7.1/scripts/ffcv/create_ffcv_datasets.py
      ```

   1. Convert the training and validation datasets.

      ```
      python /tmp/create_ffcv_datasets.py --dataset imagenet --split train --datadir /tmp/ImageNet/
      python /tmp/create_ffcv_datasets.py --dataset imagenet --split val --datadir /tmp/ImageNet/
      ```

      **Note:** The helper script output the FFCV formatted dataset files to `/tmp/imagenet_train.ffcv` and `/tmp/imagenet_val.ffcv` 
      for the training and validation data, respectively.

1. Launch the training run.

   ```
   composer -n {num_gpus} train.py -f {recipe_yaml_path} --scale_schedule_ratio {scale_schedule_ratio}
   ```

   Replace `num_gpus`, `recipe_yaml_path` and `scale_schedule_ratio` with the total number of GPU's, the recipe configuration, and the scale schedule ratio we determined in the previous section for the desired run, respectively.

   **Note:** The `Mild` and `Medium` recipes assume the training and validation data is stored at the `/tmp/imagenet_train.ffcv` and `/tmp/imagenet_val.ffcv` paths while the `Hot` recipe assumes the original ImageNet dataset is stored at the `/tmp/ImageNet` path.  The default dataset paths can be overridden, please run `composer -n {num_gpus} train.py -f {recipe_yaml_path} --help` for more detailed recipe specific configuration information.
   
   Example:
   
   ```
   composer -n 8 train.py -f recipes/resnet50_mild.yaml --scale_schedule_ratio 0.32
   ```

   The example above will train on 8 GPU's using the `Mild` recipe with a scale schedule ratio of 0.32.  You can compare your run's final Top-1 accuracy and time to train to [our result](https://explorer.mosaicml.com/imagenet?sortBy=costSameQuality&selected=fks-short-timing-r6z2-seed-17-ssr0.32&model=resnet50&cloud=all&hardware=all&algorithms=all&baseline=r50_optimized_p4d&recipe=mosaicml_mild). 