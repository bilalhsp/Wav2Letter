"""Readies the dataset to be used by dataloader.
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



# ------------------  get parser ----------------------#
def get_parser():
    parser = argparse.ArgumentParser(
        description='This is to compute and save regression results for for layers '+
            'of DNN models and neural areas',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    return parser


# ------------------  main defination ----------------------#

def main():
    train = manifest['train']
    val = manifest['val']
    test = manifest['test']
    # ready dataset for dataloader.
    prepare_dataset(manifest, train=train, val=val, test=test)


# ------------------  main function ----------------------#

if __name__ == '__main__':

    start_time = time.time()
    parser = get_parser()
    args = parser.parse_args()

    # display the arguments passed
    for arg in vars(args):
        print(f"{arg:15} : {getattr(args, arg)}")

    # calling the main function.
    main()
    elapsed_time = time.time() - start_time
    print(f"It took {elapsed_time/60:.1f} min. to run.")

