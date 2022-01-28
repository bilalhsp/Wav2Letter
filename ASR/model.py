from tabnanny import check
import torch
import torch.nn as nn
import os
import time
#import jiwer
import fnmatch
from auditory_ctx.utils.cer import cer
from auditory_ctx.utils.wer import wer

from ASR.utils import Text_Processor

class Gated_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size)->None:
        super(Gated_Conv,self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.conv2 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        A = self.conv1(x)
        B = self.conv2(x)
        B = self.sigmoid(B)
        
        out = A * B   # element-wise multiplication

        return out




class Wav2Letter2(nn.Module):
    def __init__(self) -> None:
        super(Wav2Letter2, self).__init__()
        #self.processor = Text_Processor(vocab_path=vocab_path)
        self.conv1 = Gated_Conv(in_channels=40, out_channels=100, kernel_size=3)
        self.dropout1 = nn.Dropout(0.25)
        self.gated_conv_stack = nn.Sequential(*[Gated_Conv(in_channels=100, out_channels=100, kernel_size=4+i) for i in range(15)])
        # self.conv2 = Gated_Conv(in_channels=100, out_channels=100, kernel_size=4)
        # self.conv3 = Gated_Conv(in_channels=100, out_channels=100, kernel_size=5)
        # self.conv4 = Gated_Conv(in_channels=100, out_channels=100, kernel_size=6)
        # self.conv5 = Gated_Conv(in_channels=100, out_channels=100, kernel_size=7)
        # self.conv6 = Gated_Conv(in_channels=100, out_channels=100, kernel_size=8)
        # self.conv7 = Gated_Conv(in_channels=100, out_channels=100, kernel_size=9)
        # self.conv8 = Gated_Conv(in_channels=100, out_channels=100, kernel_size=10)
        # self.conv9 = Gated_Conv(in_channels=100, out_channels=100, kernel_size=11)
        # self.conv10 = Gated_Conv(in_channels=100, out_channels=100, kernel_size=12)
        # self.conv11 = Gated_Conv(in_channels=100, out_channels=100, kernel_size=13)
        # self.conv12 = Gated_Conv(in_channels=100, out_channels=100, kernel_size=14)
        # self.conv13 = Gated_Conv(in_channels=100, out_channels=100, kernel_size=15)
        # self.conv14 = Gated_Conv(in_channels=100, out_channels=100, kernel_size=16)
        # self.conv15 = Gated_Conv(in_channels=100, out_channels=100, kernel_size=17)
        # self.conv16 = Gated_Conv(in_channels=100, out_channels=100, kernel_size=18)
        self.conv17 = Gated_Conv(in_channels=100, out_channels=375, kernel_size=19)      
        self.dropout2 = nn.Dropout(0.25)

        self.linear1 = nn.Linear(in_features=375, out_features=1000)
        self.linear2 = nn.Linear(in_features=1000, out_features=29)

        self.softmax = nn.Softmax(dim=2)
        



    def forward(self, x):
        # 'x' : (batch, n_mels, time)
        x = self.conv1(x)
        x = self.dropout1(x)
        x = self.gated_conv_stack(x)
        # x = self.conv2(x)
        # x = self.conv3(x)
        # x = self.conv4(x)
        # x = self.conv5(x)
        # x = self.conv6(x)
        # x = self.conv7(x)
        # x = self.conv8(x)
        # x = self.conv9(x)
        # x = self.conv10(x)
        # x = self.conv11(x)
        # x = self.conv12(x)
        # x = self.conv13(x)
        # x = self.conv14(x)
        # x = self.conv15(x)
        # x = self.conv16(x)
        x = self.conv17(x)
        x = self.dropout2(x)
        x = self.linear1(torch.transpose(x,1,2)) 
        out = self.linear2(x)

        # (batch, time, classes)
        
        return out
    # def decode(self, x, blank=28):
    #     torch.no_grad()
    #     x = self.forward(x)
    #     x = nn.functional.log_softmax(x, dim=2)
    #     out = torch.argmax(x, dim=2)
    #     #print(out.squeeze())
    #     pred = []
    #     batch_size = out.shape[0]
    #     for n in range(batch_size):
    #         indices = []
    #         prev_i = -1
    #         for i in out[n]:
    #             if i ==  prev_i:
    #                 continue
    #             if i == blank:
    #                 prev_i = -1
    #                 continue
    #             prev_i = i
    #             indices.append(i.item())
    #         text = self.processor.indices_to_text(indices)
    #         pred.append(self.processor.c_to_text(text))
    #     return pred



    

class SpeechRecognition(nn.Module):
    def __init__(self, vocab_path) -> None:
        super(SpeechRecognition, self).__init__()
        self.model = Wav2Letter2()
        self.processor = Text_Processor(vocab_path=vocab_path)
        self.softmax = nn.Softmax(dim=2)
        self.results_dir = './saved_results'

    def train(self, dataset, test_dataset, batch_size=128, epochs=1, learning_rate=0.9 ,momentum=0.9,
            load_weights=False, device='cpu'):
        self.model = self.model.to(device)
        
        #opt = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        opt = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum)
        loss_fn = nn.CTCLoss(blank=28)

        if load_weights:
            files = []
            for f in os.listdir(self.results_dir):
                if fnmatch.fnmatch(f, "checkpoint*.pt"):
                    files.append(f)
            files.sort()
            chkpoint = files[-1]
            checkpoint = torch.load(os.path.join(self.results_dir, chkpoint))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            opt.load_state_dict(checkpoint['opt_state_dict'])
            start_epoch = checkpoint['epoch']
            if not os.path.exits(os.path.join(self.results_dir, "results.txt")):
                with open(os.path.join(self.results_dir, "results.txt"), 'w') as f:
                    f.write(f"Epoch, Exe_time, CTCLoss \n")

        else:
            start_epoch = 0
            with open(os.path.join(self.results_dir, "results.txt"), 'w') as f:
                    f.write(f"Epoch, Exe_time, CTCLoss \n")
        train_loader = dataset.load_data(batch_size=batch_size)
        epochs = start_epoch + epochs
        loss_history = {}
        for epoch in range(start_epoch,epochs):
            start_time = time.time()
            print(f"Epoch [{epoch}]/[{epochs}]", end=": ")
            count = 0
            loss_cum = 0
            self.model.train()
            for mini_batch in train_loader:
                count += 1

                x,y, input_len, target_len = mini_batch
                x = x.to(device)
                y = y.to(device)

                opt.zero_grad()
                log_prob = self.model(x)
                log_prob = nn.functional.log_softmax(log_prob, dim=2).transpose(0,1)  #(time, batch, classes) requried shape for CTCLoss
                loss = loss_fn(log_prob, y, input_len, target_len)
                loss.backward()
                opt.step()

                loss_cum += loss.item()
            
            mini_batch_loss = loss_cum/count
            end_time = time.time()
            exe_time = (end_time-start_time)/60 
            with open(os.path.join(self.results_dir, "results.txt"), 'a') as f:
                f.write(f"{epoch}, {exe_time:.2f}, {mini_batch_loss} \n")

            if epoch %10 == 0:
                checkpoint = {'model_state_dict': self.model.state_dict(), 'opt_state_dict': opt.state_dict(), 'epoch': epoch}
                torch.save(checkpoint, os.path.join(self.results_dir, f"checkpoint_epochs_{epochs}.pt"))
                cerr, werr = self.test(test_dataset, batch_size=batch_size, shuffle=True, device=device)
                #cerr, werr = 0,0
                with open(os.path.join(self.results_dir, "results.txt"), 'a') as f:
                    f.write(f"Test set results: CER: {cerr}, WER: {werr} \n")

        

    def test(self, test_dataset, batch_size=128, shuffle=True, device='cpu'):
        test_loader = test_dataset.load_data(batch_size=batch_size, shuffle=shuffle)
        cer_cum = []
        wer_cum = []
        #count = 0
        self.model.eval()
        with torch.no_grad():
            for data in test_loader:
                #count += 1
                x,y, _, _ = data
                x = x.to(device)
                y = y.to(device)

                pred = self.decode(x)
                target = self.processor.label_to_text(y)
                
                for k in range(len(pred)):
                    cer_cum.append(cer(target[k], pred[k]))
                    wer_cum.append(wer(target[k], pred[k]))

        cerr = sum(cer_cum)/len(cer_cum)
        werr = sum(wer_cum)/len(wer_cum)

        return cerr, werr








    def decode(self, x, blank=28):
        torch.no_grad()
        x = self.model(x)
        x = nn.functional.log_softmax(x, dim=2)
        out = torch.argmax(x, dim=2)
        #print(out.squeeze())
        pred = []
        batch_size = x.shape[0]
        for n in range(batch_size):
            indices = []
            prev_i = -1
            for i in out[n]:
                if i ==  prev_i:
                    continue
                if i == blank:
                    prev_i = -1
                    continue
                prev_i = i
                indices.append(i.item())
            text = self.processor.indices_to_text(indices)
            pred.append(self.processor.c_to_text(text))
        return pred
