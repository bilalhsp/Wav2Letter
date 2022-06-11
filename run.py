
import os
from typing import Text
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader
import time
# import datasets
import fnmatch
import yaml

from ASR.dataloader.dataset import Dataset
from ASR.model import SpeechRecognition, Wav2Letter2
from ASR.utils import Text_Processor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dir = os.getcwd()
print(dir)

# Loading the config. file...!
# conf_file = 'puzzlelib_model.yaml'
#conf_file = 'pcen_puzzlelib.yaml'
#conf_file = 'exact_puzzlelib.yaml'

conf_file = 'data360.yaml'
# conf_file = 'data360_long.yaml'

manifest_file = os.path.join(dir,"conf",conf_file)
with open(manifest_file, 'r') as f:
    manifest = yaml.load(f, Loader=yaml.FullLoader)

model_param = manifest['model_param']
hyper_param = manifest['hyper_param']


#Model and other classes...!
train_data = Dataset(model_param, device=device)
#test_data = Dataset(model_param, device=device, test=True)

# model = SpeechRecognition(model_param)

# model.train(train_data, test_data, hyper_param, device=device)



# vocab_path = os.path.join(dir,model_param["vocab_dir"])
# #data_dir = "c:\\Users\\ahmedb\\projects\\speech_data"           # os.path.join(dir,'ASR', 'data')
# #data_dir = "/depot/jgmakin/data/LibriSpeech"
# #pretrained = os.path.join(data_dir, 'wav2letter_weights', 'checkpoint_epochs_280.pt')
# #Hyper parameters...!




# epochs = 500
# batch_size = 32
# learning_rate = 0.001
# momentum = 0.9

# #model paramteres for mel_spectrogram loader
# n_mel = 40
# #Model and other classes...!
# #train_data = Dataset(data_dir, vocab_path, data='training', n_mels = n_mel, device=device)
# #test_data = Dataset(data_dir, vocab_path, data='test', n_mels=n_mel, device=device)

# # paths and setup
# #results_dir = '/scratch/gilbreth/ahmedb/wav2letter/data360_adam_saved_results'
# data_dir = model_param['data_dir']
# results_dir = model_param['results_dir']
# load_weights = manifest['load_weights']

# #model paramteres used in 'puzzlelib' model
# batch_size=8
# learning_rate = 1e-5
# window_size = 400     # 0.02 s long window = sample_frequency * 0.02 =320
# window_stride = 160   #0.01 s lond window stride = sample_freq * 0.01 = 160
# window = torch.hamming_window
# channels = 201



# #Storing results...!
# mini_batch_loss = 0
# loss_cum =0 
# count = 0



# """

# End of normal code, rest is practice....!
"""
train_loader = train_data.load_data(batch_size=batch_size)
sample = next(iter(train_loader))
x,*y = sample
print("printting the dimensions of data")
print(x.shape)
#"""




# model = Wav2Letter2(vocab_path)
# model = model.to(device)
# processor = Text_Processor(vocab_path)


# opt = torch.optim.SGD(model.parameters(),lr = learning_rate,momentum=0.9)
# loss_fn = nn.CTCLoss(blank=28)

# train_loader = train_data.load_data(batch_size=batch_size)
# test_loader = test_data.load_data(batch_size=batch_size)


# print("Lengths of input and target are...!")
# itr = iter(test_loader)
# for i in range(10):
#     data = next(itr)
# #data = next(itr)
# spec, label, in_len, tar_len = data
# print(in_len)
# print(tar_len)
# #print(label)
# torch.no_grad()
# out = model(spec)
# print(out.shape)
# print(label.shape)

# # print(model.decode(spec).shape)
# # a = torch.tensor([5,2,8,0,5,5,5,5,28,4,28,4,4,3,5,2,3,4,5,6,7,8,1])
# # print(a)
# print("Loading model weights...!")
# checkpoint = torch.load(pretrained, map_location=torch.device(device))
# print(checkpoint.keys())
# model.load_state_dict(checkpoint['state_dict'])

# print("Weights loaded succesfully...!")

# print("Prediction is....!")
# print(model.decode(spec))

# print('Target is...!')
# target = processor.label_to_text(label)
# print(label.shape)
# print(target)

# print("Starting test loop............!")
# import datasets
# import jiwer
# # datasets.list_metrics()
# # cer = datasets.load_metric('cer')


# test_loader = test_data.load_data(batch_size=batch_size, shuffle=True)
# itr = iter(test_loader)
# data = next(itr)

# x,y, in_len, out_len = data

# pred = model.decode(x)
# target = processor.label_to_text(y)

# error = jiwer.cer(target, pred)
# print(f"CER is :{error}")
# wer = jiwer.wer(target, pred)

# print(f"WER is :{error}")

# results_dir = os.path.join(data_dir, 'wav2letter_weights')
# files = []
# for f in os.listdir(results_dir):
#     if fnmatch.fnmatch(f, "checkpoint*.pt"):
#         files.append(f)
# files.sort()
# print(files)


# # for epoch in range(epochs):
# #     start_time = time.time()
# #     print(f"[Epoch {epoch + 1}]/[{epochs}]:")
# #     count = 0
# #     loss_cum = 0
# #     for x, y, input_len, target_len in train_loader:
# #         print("Here...!")
# #         count += 1

# #         #load data to the GPU's (if available)
# #         x = x.to(device)
# #         y = y.to(device)

# #         #clear optimizer...!
# #         opt.zero_grad()

# #         # forward pass steps...!
# #         y_hat = model(x)        # 'y_hat' : (batch, time, classes)
# #         log_prob = nn.functional.log_softmax(y_hat, dim=2)
# #         log_prob = torch.transpose(log_prob, 0,1) # (time, batch, classes) required shape for CTC_loss
# #         loss = loss_fn(log_prob, y, input_len, target_len)     

# #         #Back-prop steps...!
# #         loss.backward()
# #         opt.step()

# #         loss_cum += loss.item()

# #     mini_batch_loss = loss_cum/count
    
# #     end_time = time.time()

    
# #     print(f"Loss/mini-batch: {mini_batch_loss}", end="")
# #     print(f"Execution time/epoch: {end_time - start_time}")


