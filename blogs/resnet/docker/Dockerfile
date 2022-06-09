ARG BASE_IMAGE

FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive

#########################
# MosaicML ResNet Recipes
#########################

# Add requirements file and install dependencies
COPY requirements.txt /mosaicml-resnet/
RUN cd /mosaicml-resnet && \
    pip install -r requirements.txt

# Add training script and recipes
COPY train.py /mosaicml-resnet/
COPY recipes /mosaicml-resnet/recipes

# Set image default working directory
WORKDIR /mosaicml-resnet