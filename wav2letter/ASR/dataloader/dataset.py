import os
import torch
import torch.nn as nn
import torchaudio
import soundfile
import pandas as pd
# import librosa
from torch.utils.data import DataLoader
# import dill
from wav2letter.ASR.utils import Text_Processor
import pytorch_lightning as pl

class Dataset():
    def __init__(self, manifest, device='cpu', test=False):
        if test:
            url = "test_url"
        else:
            url = "training_url"
        self.device = device        
        self.sample_rate = manifest["sample_rate"]
        self.hop_length = manifest["window_stride"]
        self.win_length = manifest["window_size"]
        self.n_fft = manifest["window_size"]
        self.data_dir = manifest["data_dir"]
        self.vocab_dir = manifest["vocab_dir"]
        self.download =manifest["download"]
        self.url = manifest[url]
        print("Running with these parameters..!")
        print(self.url)
        print(self.download)
        # sample_rate, n_mels, win_length and hop_length are matched with 'wav2letter2'
        self.dataset = torchaudio.datasets.LIBRISPEECH(self.data_dir, url=self.url, download = True)#self.download)
        self.text_processor = Text_Processor(vocab_path=self.vocab_dir)

        self.transform = torchaudio.transforms.Spectrogram(n_fft=self.n_fft, 
        win_length=self.win_length, hop_length=self.hop_length,
          window_fn=torch.hamming_window, power=1).to(device)
#         if data == 'training':
#             self.dataset = torchaudio.datasets.LIBRISPEECH(data_dir, url=url, download=download)
# #            self.transform = nn.Sequential(torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels, win_length=win_length,hop_length=hop_length),
# #                            torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
# #                            torchaudio.transforms.TimeMasking(time_mask_param=100)).to(device)
#         else:
#             self.dataset = torchaudio.datasets.LIBRISPEECH(data_dir, url='test-clean', download=download)
# #            self.transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels, win_length=win_length, hop_length=hop_length).to(device)


        #self.transform matched with puzzlelib implementation ....only diff being 'spectrogram' in place of 'stft'
        

    def load_data(self, batch_size, shuffle=False):
        
        loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle,
         collate_fn=self.pad_sequence)

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
            #original code before pcen...!
            wav = wav.to(self.device)
            
            # # incorporating pcen funct...!
            # wav = wav.numpy()
            # D = librosa.stft(wav, n_fft=self.win_length, hop_length=self.hop_length,
            #  win_length=self.win_length, window='hamming') 
            # spec, phase = librosa.magphase(D)
            # pcen_result = librosa.pcen(S = spec, sr=self.sample_rate, hop_length=self.hop_length)
            # print("Shape after transform and pcen....!")
            # print(pcen_result.shape)
            # pcen_result = torch.from_numpy(pcen_result)
            # spect = torch.transpose(pcen_result.squeeze(), 0,1)  #(time, n_mels)
            
            # spect.to(self.device)
            # spect = spect.type(torch.cuda.FloatTensor)
            #original code before pcen...!
            spect = torch.transpose(self.transform(wav).squeeze(), 0,1)  #(time, n_mels)
        
            # rest is the same...!
            spectrograms.append(spect) # list of '(time, n_mels)'
            label = torch.Tensor(self.text_processor.text_to_indices(trans)) #(time)
            labels.append(label) # list of (time)
            input_lengths_seq.append(spect.shape[0]//2)
            target_lengths_seq.append(label.shape[0])
        #(batch, n_mels, time) after transpose
        padded_spect = nn.utils.rnn.pad_sequence(sequences=spectrograms, batch_first=True).transpose(1,2) 
        padded_labels = nn.utils.rnn.pad_sequence(sequences=labels,batch_first=True) # (batch, time)

        return padded_spect, padded_labels, input_lengths_seq, target_lengths_seq



class LibriSpeechDataset():
    def __init__(self, manifest):
        self.manifest = pd.read_csv(manifest)
    
    def __len__(self):
        return len(self.manifest)
        
    def __getitem__(self, idx):
        audio_path = self.manifest.iloc[idx]['audio']
        trans = self.manifest.iloc[idx]['trans']
        audio, fs = soundfile.read(audio_path, always_2d=True)
        # In order to match dimensions/dtype of audio required by librosa processing...!
        # return audio.transpose(), fs, trans
        # In order to match dimensions/dtype of audio returned by pytorch loader...!
        return torch.tensor(audio.transpose(), dtype=torch.float32), fs, trans

class LSDataModule(pl.LightningDataModule):
    def __init__(self, manifest):
        super(LSDataModule, self).__init__()
      
        self.sample_rate = manifest["sample_rate"]
        self.hop_length = manifest["window_stride"]
        self.win_length = manifest["window_size"]
        self.n_fft = manifest["window_size"]
        self.data_dir = manifest["data_dir"]
        self.download =manifest["download"]
        self.num_workers = manifest["num_workers"]
        self.train_manifest = manifest["train_manifest"]
        self.val_manifest = manifest["val_manifest"]
        self.test_manifest = manifest["test_manifest"]
        # self.train_url = manifest["training_url"]
        # self.val_url = manifest["val_url"]
        # self.test_url = manifest["test_url"]
        self.batch_size=manifest["batch_size"]
        # sample_rate, n_mels, win_length and hop_length are matched with 'wav2letter2'
        self.text_processor = Text_Processor(manifest)
        self.transform = torchaudio.transforms.Spectrogram(n_fft=self.n_fft, 
        win_length=self.win_length, hop_length=self.hop_length,
          window_fn=torch.hamming_window, power=1)#.to(device)
        
    # def prepare_data(self):
    #     # torchaudio.datasets.LIBRISPEECH(self.data_dir, url=self.train_url, download = True)
    #     # torchaudio.datasets.LIBRISPEECH(self.data_dir, url=self.val_url, download = True)
    #     # torchaudio.datasets.LIBRISPEECH(self.data_dir, url=self.test_url, download = True)

    def setup(self, stage = None):
        # if stage=='fit' or stage is None:
        #     self.train_dataset = torchaudio.datasets.LIBRISPEECH(self.data_dir, url=self.train_url, download = False)
        #     self.val_dataset = torchaudio.datasets.LIBRISPEECH(self.data_dir, url=self.val_url, download = False)
        # if stage=='test' or stage is None:
        #     self.train_dataset = torchaudio.datasets.LIBRISPEECH(self.data_dir, url=self.test_url, download = False)
        # if stage=='predict' or stage is None:
        #     self.train_dataset = torchaudio.datasets.LIBRISPEECH(self.data_dir, url=self.test_url, download = False)
        
        if stage=='fit' or stage is None:
            self.train_dataset = LibriSpeechDataset(os.path.join(self.data_dir,self.train_manifest))
            self.val_dataset = LibriSpeechDataset(os.path.join(self.data_dir,self.val_manifest))
        if stage=='test' or stage is None:
            self.test_dataset = LibriSpeechDataset(os.path.join(self.data_dir,self.test_manifest))
        if stage=='predict' or stage is None:
            self.predict_dataset = LibriSpeechDataset(os.path.join(self.data_dir,self.test_manifest))


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
        shuffle=True, collate_fn=self.pad_sequence)
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
         shuffle=False, collate_fn=self.pad_sequence)
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
         shuffle=False, collate_fn=self.pad_sequence)
    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
         shuffle=False, collate_fn=self.pad_sequence)

    # def on_after_batch_transfer(self, batch, dataloader_idx):
    #     x, y, in_len, label_len = batch
    #     transform = torchaudio.transforms.Spectrogram(n_fft=self.n_fft, 
    #         win_length=self.win_length, hop_length=self.hop_length,
    #         window_fn=torch.hamming_window, power=1).to(x.device)        
    #     x = transform(x)
    #     return x, y, in_len, label_len

    def pad_sequence(self, data):
        waves = []
        labels = []
        input_lengths_seq = []
        target_lengths_seq = []
        for (wav, fs, trans, *_) in data:
            # 'wav': (1, time)
            # 'trans': string
            ###### modification........!###########
            # D = librosa.stft(wav, n_fft=self.n_fft, hop_length=self.hop_length, 
            #             win_length=self.win_length, window='hamming')
            # spect, phase = librosa.magphase(D)
            # pcenResult = librosa.pcen(S=spect, sr=self.sample_rate, hop_length=self.hop_length)
            # wav = torch.tensor(pcenResult.squeeze().transpose(), dtype=torch.float32)
            # waves.append(wav)
            #print(wav.shape)
            ###### modification ends here........!###########

            waves.append(wav.squeeze())
            label = torch.Tensor(self.text_processor.text_to_indices(trans))#, device=super().device) #(time)
            labels.append(label) # list of (time)
            in_len = int(wav.shape[1]/self.hop_length + 0.5)
            input_lengths_seq.append(in_len//2)
            target_lengths_seq.append(label.shape[0])
            # print(f"This was length before transform: {wav.shape[1]}")
            # print(f"Length after transform: {spect.shape[0]}")
        #(batch, n_mels, time) after transpose
        padded_waves = nn.utils.rnn.pad_sequence(sequences=waves, batch_first=True) #(batch, time)
        padded_labels = nn.utils.rnn.pad_sequence(sequences=labels,batch_first=True) # (batch, time)
        input_lens = torch.tensor(input_lengths_seq)
        target_lens = torch.tensor(target_lengths_seq)
        return padded_waves, padded_labels, input_lens, target_lens#, input_lengths_seq, target_lengths_seq


    def original_pad_sequence(self, data):
        spectrograms = []
        labels = []
        input_lengths_seq = []
        target_lengths_seq = []
        for (wav, fs, trans, *_) in data:
            # 'wav': (1, time)
            # 'trans': string

            wav = wav#.to(super().device)
            spect = torch.transpose(self.transform(wav).squeeze(), 0,1)  #(time, n_mels)        
            # rest is the same...!
            spectrograms.append(spect) # list of '(time, n_mels)'
            label = torch.Tensor(self.text_processor.text_to_indices(trans))#, device=super().device) #(time)
            labels.append(label) # list of (time)
            input_lengths_seq.append(spect.shape[0]//2)
            target_lengths_seq.append(label.shape[0])
        #(batch, n_mels, time) after transpose
        padded_spect = nn.utils.rnn.pad_sequence(sequences=spectrograms, batch_first=True).transpose(1,2) 
        padded_labels = nn.utils.rnn.pad_sequence(sequences=labels,batch_first=True) # (batch, time)
        input_lens = torch.tensor(input_lengths_seq)
        target_lens = torch.tensor(target_lengths_seq)
        return padded_spect, padded_labels, input_lens, target_lens#, input_lengths_seq, target_lengths_seq

