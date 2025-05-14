"""Downloads the LibriSpeech datasets as per urls mentioned in the config file,
and readies the dataset to be used by dataloader.
If dataset is to be re-downloaded, as will be the case most of the time, set 
'force_redownload flag to be true in the config file. 
"""

# built-in
import os
import shutil
import argparse
import time

# third part
import torchaudio

# local
from wav2letter.prepareLibriSpeech import *
from wav2letter import manifest

import logging



# ------------------  get parser ----------------------#
def get_parser():
    parser = argparse.ArgumentParser(
        description='This is to download LibriSpeech dataset and '+
            'create manifest file that will be used by the data loader.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-tr','--training', dest='training', action='store_true', default=False,
        help="Specify if training data to be downloaded."
    )
    parser.add_argument(
        '-v','--validation', dest='validation', action='store_true', default=False,
        help="Specify if validation data to be downloaded."
    )
    parser.add_argument(
        '-ts','--test', dest='test', action='store_true', default=False,
        help="Specify if validation data to be downloaded."
    )
    parser.add_argument(
        '-c','--clear', dest='clear', action='store_true', default=False,
        help="Specify if existing directory structure needs to be cleared."
    )

    return parser


# ------------------  download librispeech ----------------------#

def download_librispeech(args):

    data_dir = manifest['data_dir']

    # Create and empty parent directory
    if args.clear:
        # remove an existing tree of folders
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)
        # create parent folder for the datadirectory.
        os.makedirs(data_dir)
        logging.info("Successfully created an empty parent directory.")

    train = args.training
    val = args.validation
    test = args.test

    training_urls = [
        'train-clean-100',
        # 'train-clean-360',
        # 'train-other-500',
        ]
    val_urls = ['dev-clean', 'dev-other']
    test_urls = ['test-clean', 'test-other']

    # Download data to be used.
    if train:
        for url in training_urls:
            logging.info("Creating dataset for url: {}".format(url))
            data = torchaudio.datasets.LIBRISPEECH(manifest['data_dir'], url=url, download=True)
    if val:
        for url in val_urls:
            logging.info("Creating dataset for url: {}".format(url))
            data = torchaudio.datasets.LIBRISPEECH(manifest['data_dir'], url=url, download=True)
    if test:
        for url in test_urls:
            logging.info("Creating dataset for url: {}".format(url))
            data = torchaudio.datasets.LIBRISPEECH(manifest['data_dir'], url=url, download=True)

# # ready dataset for dataloader.
# prepare_dataset(manifest, train=train, val=val, test=test)


# ------------------  main function ----------------------#

if __name__ == '__main__':

    start_time = time.time()
    parser = get_parser()
    args = parser.parse_args()

    # display the arguments passed
    for arg in vars(args):
        print(f"{arg:15} : {getattr(args, arg)}")

    download_librispeech(args)
    elapsed_time = time.time() - start_time
    print(f"It took {elapsed_time/60:.1f} min. to run.")