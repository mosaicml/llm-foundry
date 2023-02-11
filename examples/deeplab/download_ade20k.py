# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""ADE20K Semantic segmentation and scene parsing dataset.

Please refer to the `ADE20K dataset <https://groups.csail.mit.edu/vision/datasets/ADE20K/>`_ for more details about this
dataset.
"""

import argparse

import torchvision

parser = argparse.ArgumentParser()

parser.add_argument('--path',
                    help='ADE20k Download directory.',
                    type=str,
                    default='./ade20k')

args = parser.parse_args()

ADE20K_URL = 'http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip'
ADE20K_FILE = 'ADEChallengeData2016.zip'


def main():
    torchvision.datasets.utils.download_and_extract_archive(
        url=ADE20K_URL,
        download_root=args.path,
        filename=ADE20K_FILE,
        remove_finished=True)


if __name__ == '__main__':
    main()
