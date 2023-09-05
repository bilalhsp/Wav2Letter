# built-in
import os
import shutil

# third part
import torchaudio

# local
from wav2letter.prepareLibriSpeech import *
from wav2letter import config 



"""Downloads the LibriSpeech datasets as per urls mentioned in the config file,
and readies the dataset to be used by dataloader.
If dataset is to be re-downloaded, as will be the case most of the time, set 
'force_redownload flag to be true in the config file. 
"""

data_dir = config['data_dir']

# Create and empty parent directory
if config['force_redownload']:
    # remove an existing tree of folders
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    # create parent folder for the datadirectory.
    os.makedirs(data_dir)
    print("Successfully created an empty parent directory.")

# Download data to be used.
for url in config['training_url']:
    data = torchaudio.datasets.LIBRISPEECH(config['data_dir'], url=url, download=True)
for url in config['val_url']:
    data = torchaudio.datasets.LIBRISPEECH(config['data_dir'], url=url, download=True)
for url in config['test_url']:
    data = torchaudio.datasets.LIBRISPEECH(config['data_dir'], url=url, download=True)

# ready dataset for dataloader.
prepare_dataset(config, train=True, val=True, test=True)