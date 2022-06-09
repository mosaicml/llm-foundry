# Base image to use for the docker build.
BASE_IMAGE ?= 'mosaicml/pytorch_vision:resnet_recipes_base'

# Resulting docker tag for the composer dockerimage
IMAGE_TAG ?= mosaicml/pytorch_vision:resnet50_recipes

.PHONY: resnet 

resnet:
	docker build -t $(IMAGE_TAG) \
		--build-arg BASE_IMAGE=$(BASE_IMAGE) \
		-f Dockerfile \
		..
