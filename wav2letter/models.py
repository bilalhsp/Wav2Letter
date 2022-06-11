from tabnanny import check
import torch
import torch.nn as nn
import torchaudio
import pytorch_lightning as pl
import os
import time
import yaml
#import jiwer
import fnmatch
from statistics import mean
from wav2letter.utils.cer import cer
from wav2letter.utils.wer import wer
from wav2letter.processors import Text_Processor


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, dropout, dilation=1):
        super(conv_block, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #print("Conv forward.....")
        
        x = self.conv(x)
        x = self.batch_norm(x)
        x = torch.clamp(x, min=0.0, max=20.0)
        # x = self.activation(x)
        out = self.dropout(x)
        #print(out.shape)
        return out
         

class Gated_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1)->None:
        super(Gated_Conv,self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.conv2 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.sigmoid = nn.Sigmoid()
        self.layer_norm = nn.GroupNorm(in_channels, in_channels)
        self.batch_norm = nn.BatchNorm1d(in_channels)
	#self.layer_norm = Layer_Norm(in_channels)

    def forward(self, x):
        #x = self.layer_norm(x)
        x = self.batch_norm(x)
        #print("normalizing in GLU")
        #print(x.shape)
        #std, mean = torch.std_mean(x, dim=2, keepdim=True)
        #x = (x-mean)/std
        A = self.conv1(x)
        B = self.conv2(x)
        B = self.sigmoid(B)
        
        out = A * B   # element-wise multiplication

        return out

class Layer_Norm(nn.Module):
    def __init__(self, feats):
        super(Layer_Norm, self).__init__()
        self.layer_norm = nn.LayerNorm(feats)

    def forward(self, x):
        #(batch, ch, feats, time)
        x = x.transpose(2,1).contiguous()
        x = self.layer_norm(x)
        return x.transpose(2,1).contiguous()


class Wav2Letter2(nn.Module):
    def __init__(self, n_mels=40) -> None:
        super(Wav2Letter2, self).__init__()
        #self.processor = Text_Processor(vocab_path=vocab_path)
        #self.layer_norm = Layer_Norm(40) 
        #self.conv1 = Gated_Conv(in_channels=n_mels, out_channels=200, kernel_size=13, padding=6, stride=2)
        #self.dropout1 = nn.Dropout(0.20)
        #self.gated_conv_stack = nn.Sequential(*[Gated_Conv(in_channels=200, out_channels=200, kernel_size=14+i, padding=7+i) for i in range(15)])
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
        #self.conv17 = Gated_Conv(in_channels=200, out_channels=750, kernel_size=29, padding=14)      
        #self.dropout2 = nn.Dropout(0.25)

        #self.linear1 = nn.Linear(in_features=750, out_features=1500)
        #self.linear2 = nn.Linear(in_features=1500, out_features=29)

        #self.softmax = nn.Softmax(dim=2)

        # implementation using conv_block...!
        #self.conv_block_stack = nn.Sequential(
        self.conv1 = conv_block(in_channels=n_mels,  out_channels=256, kernel_size=11,  padding=5, stride=2,  dropout=0.2)

        self.conv2 = conv_block(in_channels=256,  out_channels=256, kernel_size=11,  padding=5, stride=1,  dropout=0.2)
        self.conv3 = conv_block(in_channels=256,  out_channels=256, kernel_size=11,  padding=5, stride=1,  dropout=0.2)
        self.conv4 = conv_block(in_channels=256,  out_channels=256, kernel_size=11,  padding=5, stride=1,  dropout=0.2)

        self.conv5 = conv_block(in_channels=256,  out_channels=384, kernel_size=13,  padding=6, stride=1,  dropout=0.2)
        self.conv6 = conv_block(in_channels=384,  out_channels=384, kernel_size=13,  padding=6, stride=1,  dropout=0.2)
        self.conv7 = conv_block(in_channels=384,  out_channels=384, kernel_size=13,  padding=6, stride=1,  dropout=0.2)

        self.conv8 = conv_block(in_channels=384,  out_channels=512, kernel_size=17,  padding=8, stride=1,  dropout=0.2)
        self.conv9 = conv_block(in_channels=512,  out_channels=512, kernel_size=17,  padding=8, stride=1,  dropout=0.2)
        self.conv10 = conv_block(in_channels=512,  out_channels=512, kernel_size=17,  padding=8, stride=1,  dropout= 0.2)
 
        self.conv11 = conv_block(in_channels=512,  out_channels=640, kernel_size=21,  padding=10, stride=1,  dropout=0.3)
        self.conv12 = conv_block(in_channels=640,  out_channels=640, kernel_size=21,  padding=10, stride=1,  dropout=0.3)
        self.conv13 = conv_block(in_channels=640,  out_channels=640, kernel_size=21,  padding=10, stride=1,  dropout=0.3)

        self.conv14 = conv_block(in_channels=640,  out_channels=768, kernel_size=25,  padding=12, stride=1,  dropout=0.3)
        self.conv15 = conv_block(in_channels=768,  out_channels=768, kernel_size=25,  padding=12, stride=1,  dropout=0.3)
        self.conv16 = conv_block(in_channels=768,  out_channels=768, kernel_size=25,  padding=12, stride=1,  dropout=0.3)

        self.conv17 = conv_block(in_channels=768, out_channels=896, kernel_size=29,  padding=28, stride=1,  dropout=0.4, dilation=2)


        self.conv18 = conv_block(in_channels=896,  out_channels=1024, kernel_size=1,  padding=0, stride=1,  dropout=0.4)
        self.conv19 = conv_block(in_channels=1024,  out_channels=29, kernel_size=1,  padding=0, stride=1,  dropout=0.0)
                 
        

    def forward(self, x):

        #conv_block ...!
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)
        out = self.conv19(x)
        out = torch.transpose(out, 1,2)
        # earlier implementation...
        # 'x' : (batch, n_mels, time)
        #x = x.unsqueeze(1)
        #x = self.layer_norm(x)
        #print("Just before normalizing")
        #print(x.shape)
        #std, mean = torch.std_mean(x, dim=2, keepdim=True)
        #x = (x-mean)/std
        #x = self.conv1(x)
        #x = self.dropout1(x)
        #x = self.gated_conv_stack(x)
        
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
        #x = self.conv17(x)
        #x = self.dropout2(x)
        #print("Shape after the stack")
        #x = torch.flatten(x, start_dim=1, end_dim=2)
        #print(x.shape)
        #x = self.linear1(torch.transpose(x,1,2)) 
        #out = self.linear2(x)

        # (batch, time, classes)
        
        return out
    # def decode(self, x, blank=28):
    #     torch.no_grad()
        #return out
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
    def __init__(self, manifest) -> None:
        super(SpeechRecognition, self).__init__()
        self.results_dir = manifest["results_dir"]
        self.vocab_dir = manifest["vocab_dir"]
        self.channels = manifest["channels"]
        
        self.model = Wav2Letter2(self.channels)
        self.processor = Text_Processor(vocab_path=self.vocab_dir)
        self.softmax = nn.Softmax(dim=2)
        

    def train(self, dataset, test_dataset, manifest, device='cpu'):
        self.model = self.model.to(device)
        epochs = manifest["epochs"]
        batch_size = manifest["batch_size"]
        learning_rate = manifest["learning_rate"]
        momentum = manifest["momentum"]
        load_weights = manifest["load_weights"]
        
        opt = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        #opt = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum)
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
            if not os.path.exists(os.path.join(self.results_dir, "results.txt")):
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
            
            if epoch %10==0:  
                checkpoint = {'model_state_dict': self.model.state_dict(), 'opt_state_dict': opt.state_dict(), 'epoch': epoch}
                torch.save(checkpoint, os.path.join(self.results_dir, f"checkpoint_epochs_{epoch:03d}.pt"))
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



class LitWav2Letter(pl.LightningModule):
    def __init__(self) -> None:
        super(LitWav2Letter, self).__init__()
        # Loading the config file..!
        manifest_file = os.path.join(os.path.dirname(__file__),'conf','lightning.yaml')
        with open(manifest_file, 'r') as f:
            manifest = yaml.load(f, Loader=yaml.FullLoader)

        self.model_name = "wav2letter"
        self.manifest = manifest
        self.results_dir = manifest["results_dir"]
        self.channels = manifest["channels"]
        self.hop_length = manifest["window_stride"]
        self.win_length = manifest["window_size"]
        self.n_fft = manifest["window_size"]
        self.lr = manifest["learning_rate"]
        
       #self.model = Wav2Letter2(self.channels)
        self.processor = Text_Processor()
        self.loss_fn = nn.CTCLoss(blank=28).to(self.device)
        self.transform = torchaudio.transforms.Spectrogram(n_fft=self.n_fft, 
            win_length=self.win_length, hop_length=self.hop_length,
            window_fn=torch.hamming_window, power=1).to(self.device)
        #self.softmax = nn.Softmax(dim=2)

        self.conv1 = conv_block(in_channels=self.channels,  out_channels=256, kernel_size=11,  padding=5, stride=2,  dropout=0.2)
        self.conv2 = conv_block(in_channels=256,  out_channels=256, kernel_size=11,  padding=5, stride=1,  dropout=0.2)
        self.conv3 = conv_block(in_channels=256,  out_channels=256, kernel_size=11,  padding=5, stride=1,  dropout=0.2)
        self.conv4 = conv_block(in_channels=256,  out_channels=256, kernel_size=11,  padding=5, stride=1,  dropout=0.2)

        self.conv5 = conv_block(in_channels=256,  out_channels=384, kernel_size=13,  padding=6, stride=1,  dropout=0.2)
        self.conv6 = conv_block(in_channels=384,  out_channels=384, kernel_size=13,  padding=6, stride=1,  dropout=0.2)
        self.conv7 = conv_block(in_channels=384,  out_channels=384, kernel_size=13,  padding=6, stride=1,  dropout=0.2)

        self.conv8 = conv_block(in_channels=384,  out_channels=512, kernel_size=17,  padding=8, stride=1,  dropout=0.2)
        self.conv9 = conv_block(in_channels=512,  out_channels=512, kernel_size=17,  padding=8, stride=1,  dropout=0.2)
        self.conv10 = conv_block(in_channels=512,  out_channels=512, kernel_size=17,  padding=8, stride=1,  dropout= 0.2)
 
        self.conv11 = conv_block(in_channels=512,  out_channels=640, kernel_size=21,  padding=10, stride=1,  dropout=0.3)
        self.conv12 = conv_block(in_channels=640,  out_channels=640, kernel_size=21,  padding=10, stride=1,  dropout=0.3)
        self.conv13 = conv_block(in_channels=640,  out_channels=640, kernel_size=21,  padding=10, stride=1,  dropout=0.3)

        self.conv14 = conv_block(in_channels=640,  out_channels=768, kernel_size=25,  padding=12, stride=1,  dropout=0.3)
        self.conv15 = conv_block(in_channels=768,  out_channels=768, kernel_size=25,  padding=12, stride=1,  dropout=0.3)
        self.conv16 = conv_block(in_channels=768,  out_channels=768, kernel_size=25,  padding=12, stride=1,  dropout=0.3)

        self.conv17 = conv_block(in_channels=768, out_channels=896, kernel_size=29,  padding=28, stride=1,  dropout=0.4, dilation=2)


        self.conv18 = conv_block(in_channels=896,  out_channels=1024, kernel_size=1,  padding=0, stride=1,  dropout=0.4)
        self.conv19 = conv_block(in_channels=1024,  out_channels=29, kernel_size=1,  padding=0, stride=1,  dropout=0.0)


    def forward(self, x):
        #conv_block ...!

        #### librosa modification...!####
        # x = torch.transpose(x,1,2)
        # print(x.shape)
        #### librosa modification...!####
        x = self.transform(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)
        out = self.conv19(x)
        out = torch.transpose(out, 1,2)

        return out
    
    def configure_optimizers(self):
        # learning_rate = 1.0e-4
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return optimizer

    def training_step(self, train_batch, batch_idx): 
        
        x,y, input_len, target_len = train_batch
        # print("\n Training step in the device: ")
        # print(self.device)
        # print(f"batch_size: {x.shape[0]}")

        log_prob = self.forward(x)
        log_prob = nn.functional.log_softmax(log_prob, dim=2).transpose(0,1)  #(time, batch, classes) requried shape for CTCLoss
        loss = self.loss_fn(log_prob, y, input_len, target_len)

        # self.log('train_loss', loss, on_step=False, on_epoch=True)
        ####changes made to the training_step...!
        logs = {'train_loss': loss}
        output = {
                'loss': loss,
                'log': logs,
                }
        return output

    def validation_step(self, val_batch, batch_idx):      
        x,y, input_len, target_len = val_batch
        # print(f"batch_size: {x.shape[0]}")

        log_prob = self.forward(x)
        log_prob = nn.functional.log_softmax(log_prob, dim=2).transpose(0,1)  #(time, batch, classes) requried shape for CTCLoss
        loss = self.loss_fn(log_prob, y, input_len, target_len)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        pred = self.decode(x)
        target = self.processor.label_to_text(y)
        cer_cum = []
        wer_cum = []
        for k in range(len(pred)):
            cer_cum.append(cer(target[k], pred[k]))
            wer_cum.append(wer(target[k], pred[k]))
        cerr = sum(cer_cum)/len(cer_cum)
        werr = sum(wer_cum)/len(wer_cum)
        # self.log('val_cerr', cerr, on_step=False, on_epoch=True)
        # self.log('val_werr', werr, on_step=False, on_epoch=True)

        output = {'loss': loss, 'cer':cerr, 'wer': werr}
        return output
        # return loss, cerr, werr

    def training_epoch_end(self,outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Loss/Train", avg_loss, self.current_epoch)
        if self.current_epoch==1:
            sample_input = torch.rand(32, 10000)
            self.logger.experiment.add_graph(LitWav2Letter(self.manifest), sample_input)
        # epoch_dictionary = {'loss': avg_loss}  
        # return epoch_dictionary

    def validation_epoch_end(self,outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_cer = mean([x['cer'] for x in outputs])
        avg_wer = mean([x['wer'] for x in outputs])
        self.logger.experiment.add_scalar("Loss/Val", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("CER/Val", avg_cer, self.current_epoch)
        self.logger.experiment.add_scalar("WER/Val", avg_wer, self.current_epoch)
        # epoch_dictionary = {'loss': avg_loss}
        # return epoch_dictionary


    def decode(self, x, blank=28):
        torch.no_grad()
        x = self.forward(x)
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


