

# ================================================================
# Help
# ================================================================

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("make %-8s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

.PHONY: help
help:  ## Print this help message
	@echo "Here are the available commands:"
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)


# ================================================================
# Main targets
# ================================================================

.PHONY: style
style:  ## Apply autoformating and run style checks via pre-commit
	pre-commit run --all-files

.PHONY: lint
lint:  ## Apply autoformating and run style checks via pre-commit
	@echo "================================ Linting BERT examples"
	bash scripts/lint_subdirectory.sh bert
	@echo "================================ Linting Resnet CIFAR examples"
	bash scripts/lint_subdirectory.sh resnet_cifar
	@echo "================================ Linting DeepLab examples"
	bash scripts/lint_subdirectory.sh deeplab
	@echo "================================ Linting LLM examples"
	bash scripts/lint_subdirectory.sh llm
	@echo "================================ Linting ResNet ImageNet examples"
	bash scripts/lint_subdirectory.sh resnet_imagenet
	@echo "================================ Linting Stable Diffusion Finetuning examples"
	bash scripts/lint_subdirectory.sh stable-diffusion

# we don't test the BERT examples since there are no tests yet...
.PHONY: test
test:  ## Run all the tests
	@echo "================================ Testing BERT examples"
	bash scripts/test_subdirectory.sh bert
	@echo "================================ Testing ResNet CIFAR examples"
	bash scripts/test_subdirectory.sh resnet_cifar
	@echo "================================ Testing DeepLab examples"
	bash scripts/test_subdirectory.sh deeplab
	@echo "================================ Testing LLM examples"
	bash scripts/test_subdirectory.sh llm
	@echo "================================ Testing ResNet ImageNet examples"
	bash scripts/test_subdirectory.sh resnet_imagenet
	@echo "================================ Testing Stable Diffusion Finetuning examples"
	bash scripts/test_subdirectory.sh stable-diffusion
