import os
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader

from ASR.utils import Text_Processor

class Data_Loader():
    def __init__(self, data_dir, vocab_path, sample_rate=16000,n_mels=40,win_length=400, hop_length=160):
        # sample_rate, n_mels, win_length and hop_length are matched with 'wav2letter2'
         
        self.train_dataset = torchaudio.datasets.LIBRISPEECH(data_dir)
        self.text_processor = Text_Processor(vocab_path=vocab_path)
        self.transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,n_mels=n_mels,win_length=win_length, hop_length=hop_length)

    def load_data(self, batch_size, train=True):
        if train:
            loader = DataLoader(self.train_dataset, batch_size=batch_size, collate_fn=lambda x: self.pad_sequence(x))

        return loader 


    def pad_sequence(self, data):
    #def pad_seq(data):
        spectrograms = []
        labels = []
        input_lengths_seq = []
        target_lengths_seq = []
        for (wav, fs, trans, *_) in data:
            # 'wav': (1, time)
            # 'trans': string 
            spect = torch.transpose(self.transform(wav).squeeze(), 0,1)  #(time, n_mels)
            spectrograms.append(spect) # list of '(time, n_mels)'
            label = torch.Tensor(self.text_processor.text_to_indices(trans)) #(time)
            labels.append(label) # list of (time)
            input_lengths_seq.append(spect.shape[0]//2)
            target_lengths_seq.append(label.shape[0])

        padded_spect = nn.utils.rnn.pad_sequence(sequences=spectrograms, batch_first=True).transpose(1,2) #(batch, n_mels, time) after transpose
        padded_labels = nn.utils.rnn.pad_sequence(sequences=labels,batch_first=True) # (batch, time)

        return padded_spect, padded_labels, input_lengths_seq, target_lengths_seq