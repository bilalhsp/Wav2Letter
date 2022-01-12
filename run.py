import os
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader
import time

from ASR.dataloader.dataloader import Data_Loader
from ASR.model import Wav2Letter2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dir = os.getcwd()
print(dir)
vocab_path = os.path.join(dir,"ASR","vocab.letters.28")
data_dir = "c:\\Users\\ahmedb\\projects\\speech_data"           # os.path.join(dir,'ASR', 'data')

#Hyper parameters...!
epochs = 1
batch_size = 5 
learning_rate = 0.01

#Storing results...!
mini_batch_loss = 0
loss_cum =0 
count = 0

#Model and other classes...!
data_loader = Data_Loader(data_dir, vocab_path)
model = Wav2Letter2()
model = model.to(device)


opt = torch.optim.SGD(model.parameters(),lr = learning_rate,momentum=0.9)
loss_fn = nn.CTCLoss(blank=28)

train_loader = data_loader.load_data(batch_size=batch_size)

print("Trainin starts...!")

for epoch in range(epochs):
    start_time = time.time()
    print(f"[Epoch {epoch + 1}]/[{epochs}]:")
    count = 0
    loss_cum = 0
    for x, y, input_len, target_len in train_loader:
        print("Here...!")
        count += 1

        #load data to the GPU's (if available)
        x = x.to(device)
        y = y.to(device)

        #clear optimizer...!
        opt.zero_grad()

        # forward pass steps...!
        y_hat = model(x)        # 'y_hat' : (batch, time, classes)
        log_prob = nn.functional.log_softmax(y_hat, dim=2)
        log_prob = torch.transpose(log_prob, 0,1) # (time, batch, classes) required shape for CTC_loss
        loss = loss_fn(log_prob, y, input_len, target_len)     

        #Back-prop steps...!
        loss.backward()
        opt.step()

        loss_cum += loss.item()

    mini_batch_loss = loss_cum/count
    
    end_time = time.time()

    
    print(f"Loss/mini-batch: {mini_batch_loss}", end="")
    print(f"Execution time/epoch: {end_time - start_time}")


