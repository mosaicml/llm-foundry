ARG BASE_IMAGE
FROM $BASE_IMAGE

ARG BRANCH_NAME
ARG DEP_GROUPS
ARG TE_COMMIT
ARG KEEP_FOUNDRY="false"

ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 8.7 8.9 9.0"

RUN echo "Starting build with BRANCH_NAME=${BRANCH_NAME}, DEP_GROUPS=${DEP_GROUPS}, TE_COMMIT=${TE_COMMIT}, KEEP_FOUNDRY=${KEEP_FOUNDRY}"

# Check for changes in setup.py.
ADD https://raw.githubusercontent.com/mosaicml/llm-foundry/$BRANCH_NAME/setup.py setup.py
RUN rm setup.py

# Install TransformerEngine
RUN NVTE_FRAMEWORK=pytorch CMAKE_BUILD_PARALLEL_LEVEL=4 MAX_JOBS=4 pip install git+https://github.com/NVIDIA/TransformerEngine.git@$TE_COMMIT

RUN echo "Cloning llm-foundry repository"
RUN git clone -b $BRANCH_NAME https://github.com/mosaicml/llm-foundry.git

RUN echo "Installing llm-foundry"
RUN pip install --no-cache-dir "./llm-foundry${DEP_GROUPS}"

RUN echo "KEEP_FOUNDRY value: ${KEEP_FOUNDRY}"

# Conditionally uninstall llm-foundry and remove its directory
RUN if [ "$KEEP_FOUNDRY" != "true" ]; then \
      echo "Uninstalling llm-foundry"; \
      pip uninstall -y llm-foundry && \
      rm -rf llm-foundry; \
    else \
      echo "Keeping llm-foundry installed"; \
    fi

# Add a final check to see if llm-foundry is installed
RUN pip list | grep llm-foundry || echo "llm-foundry not found in pip list"

# Print the contents of the directory where llm-foundry was cloned
RUN ls -la /llm-foundry || echo "llm-foundry directory not found"

# Try to import llm-foundry in Python
RUN python -c "import llm_foundry; print('llm-foundry successfully imported')" || echo "Failed to import llm-foundry"

RUN echo "Build process completed"