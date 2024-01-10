# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile

from pytest import fixture


@fixture
def gcs_account_credentials():
    """Mocked GCS Credentials for service level account."""
    os.environ['GCS_KEY'] = '🗝️'
    os.environ['GCS_SECRET'] = '🤫'
    yield
    del os.environ['GCS_KEY']
    del os.environ['GCS_SECRET']


@fixture
def uc_account_credentials():
    """Mocked UC Credentials for service level account."""
    os.environ['DATABRICKS_HOST'] = '⛵️'
    os.environ['DATABRICKS_TOKEN'] = '😶‍🌫️'
    yield
    del os.environ['DATABRICKS_HOST']
    del os.environ['DATABRICKS_TOKEN']


@fixture
def oci_temp_file():
    """Mocked OCI settings file."""
    file = tempfile.NamedTemporaryFile()
    os.environ['OCI_CONFIG_FILE'] = file.name

    yield

    file.close()
    del os.environ['OCI_CONFIG_FILE']
