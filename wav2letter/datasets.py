import os
import torch
import torch.nn as nn
import torchaudio
import soundfile
import pandas as pd
import numpy as np
import yaml
# import librosa
from torch.utils.data import DataLoader
# import dill
from wav2letter.processors import Text_Processor
from wav2letter import manifest
import pytorch_lightning as pl


def generate_lp_bounded_vector(n, eps, p=2):

    random_vector = np.random.uniform(-1, 1, n)
    lp_norm = np.linalg.norm(random_vector, ord=p)

    # Normalize the vector to have Lp norm equal to B
    bounded_vector = (random_vector/lp_norm )*eps

    return bounded_vector


class LibriSpeechDataset():
    def __init__(self, manifest, adv_robust=False, norm_bound = 0.1):
        self.manifest = pd.read_csv(manifest)
        self.adv_robust = adv_robust
        self.norm_bound = norm_bound
    
    def __len__(self):
        return len(self.manifest)
        
    def __getitem__(self, idx):
        audio_path = self.manifest.iloc[idx]['audio']
        trans = self.manifest.iloc[idx]['trans']
        audio, fs = soundfile.read(audio_path, always_2d=True)
        audio = audio.squeeze()
        if self.adv_robust:
            n = audio.shape[0]
            delta = generate_lp_bounded_vector(n, self.norm_bound, p=2)
            # print(audio.shape)
            # print(delta.shape)
            audio = audio + delta
        # In order to match dimensions/dtype of audio required by librosa processing...!
        # return audio.transpose(), fs, trans
        # In order to match dimensions/dtype of audio returned by pytorch loader...!
        return torch.tensor(np.expand_dims(audio, axis=0), dtype=torch.float32), fs, trans

class DataModuleRF(pl.LightningDataModule):
    def __init__(self):
        super(DataModuleRF, self).__init__()
        self.sample_rate = manifest["sample_rate"]
        self.hop_length = manifest["window_stride"]

        self.data_dir = manifest["data_dir"]
        self.download =manifest["download"]
        self.num_workers = manifest["num_workers"]
        self.train_manifest = manifest["train_manifest"]
        self.val_manifest = manifest["val_manifest"]
        self.test_manifest = manifest["test_manifest"]
        self.batch_size=manifest["batch_size"]
        self.adv_robust = manifest['adv_robust']
        self.norm_bound = manifest['norm_bound']
        self.persistant_workers = True if self.num_workers > 0 else False
        # sample_rate, n_mels, win_length and hop_length are matched with 'wav2letter2'
        self.text_processor = Text_Processor()
        
    def setup(self, stage = None):
        
        if stage=='fit' or stage is None:
            self.train_dataset = LibriSpeechDataset(os.path.join(self.data_dir,self.train_manifest),
                                                    self.adv_robust, self.norm_bound)
            self.val_dataset = LibriSpeechDataset(os.path.join(self.data_dir,self.val_manifest),
                                                  self.adv_robust, self.norm_bound)
        if stage=='test' or stage is None:
            self.test_dataset = LibriSpeechDataset(os.path.join(self.data_dir,self.test_manifest),
                                                   self.adv_robust, self.norm_bound)
        if stage=='predict' or stage is None:
            self.predict_dataset = LibriSpeechDataset(os.path.join(self.data_dir,self.test_manifest),
                                                      self.adv_robust, self.norm_bound)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
        shuffle=True, collate_fn=self.pad_sequence,
        pin_memory=True, persistent_workers=self.persistant_workers
        )
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
        shuffle=False, collate_fn=self.pad_sequence,
        pin_memory=True, persistent_workers=self.persistant_workers
        )
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
         shuffle=False, collate_fn=self.pad_sequence,
         pin_memory=True, persistent_workers=self.persistant_workers  
        )
    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
         shuffle=False, collate_fn=self.pad_sequence,
         pin_memory=True, persistent_workers=self.persistant_workers 
        )


    def pad_sequence(self, data):
        waves = []
        labels = []
        input_lengths_seq = []
        target_lengths_seq = []
        for (wav, fs, trans, *_) in data:

            waves.append(wav.squeeze())
            label = torch.Tensor(self.text_processor.text_to_indices(trans))#, device=super().device) #(time)
            labels.append(label) # list of (time)
            in_len = int(wav.shape[1]/self.hop_length + 0.5)
            input_lengths_seq.append(in_len//2)
            target_lengths_seq.append(label.shape[0])

        padded_waves = nn.utils.rnn.pad_sequence(sequences=waves, batch_first=True) #(batch, time)
        padded_labels = nn.utils.rnn.pad_sequence(sequences=labels,batch_first=True) # (batch, time)
        input_lens = torch.tensor(input_lengths_seq)
        target_lens = torch.tensor(target_lengths_seq)
        return padded_waves, padded_labels, input_lens, target_lens#, input_lengths_seq, target_lengths_seq


