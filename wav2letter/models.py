from tabnanny import check
from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
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


################################################################################################
# ####################              Wave2letter_spect         ############################
###############      Untrained Study ....
################################################################################################
class conv_block_spect(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, dropout, dilation=1):
        super(conv_block_spect, self).__init__()
        self.batch_norm = nn.BatchNorm1d(in_channels)
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation)
        
        self.activation = nn.Hardtanh()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #print("Conv forward.....")
        x = self.batch_norm(x)
        x = self.conv(x)
        # x = torch.clamp(x, min=0.0, max=20.0)
        x = self.activation(x)
        out = self.dropout(x)
        #print(out.shape)
        return out

### Modified for receptive fields#######

class Wav2LetterSpect(pl.LightningModule):
    def __init__(self, model_config = 'config_rf.yaml') -> None:
        super(Wav2LetterSpect, self).__init__()
        # Loading the config file..!
        manifest_file = os.path.join(os.path.dirname(__file__), 'conf', model_config)
        with open(manifest_file, 'r') as f:
            self.manifest = yaml.load(f, Loader=yaml.FullLoader)

        self.model_name = "wav2letter_spect"
        self.results_dir = self.manifest["results_dir"]
        self.channels = 80
        self.lr = self.manifest["learning_rate"]
        
       #self.model = Wav2Letter2(self.channels)
        self.processor = Text_Processor()
        self.loss_fn = nn.CTCLoss(blank=28).to(self.device)

        self.validation_step_outputs = []
        self.training_step_outputs = []
        num_units = 1024
        # rf = 65
        # rf = 145
        # rf = 225
        rf = 785

        print(f"Creating w2l_spect for RF: {rf} and units: {num_units}")
        if rf==65:
            kernels = [5,1,1, 1,1, 1,1,1, 1,1,1,1]      
            strides = [2,1,1, 1,1, 1,1,1, 1,1,1,1]
        elif rf==145:
            kernels = [5,5,1, 1,1, 1,1,1, 1,1,1,1]
            strides = [2,1,1, 1,1, 1,1,1, 1,1,1,1]
        elif rf==225:
            kernels = [5,5,5, 1,1, 1,1,1, 1,1,1,1]
            strides = [2,1,1, 1,1, 1,1,1, 1,1,1,1]
        else:
            kernels = [5,5,5, 5,5, 5,5,5, 3,3,3,3]  
            strides = [2,1,1, 1,1, 1,1,1, 1,1,1,1]

        # paddings = [2,2,2, 2,2, 2,2,2, 1,1,1,1]
        dropouts = [0.2,0.2,0.2, 0.2,0.2, 0.2,0.2,0.2, 0.2,0.2,0.3,0.3] 

        idx = 0
        self.conv1 = conv_block_spect(
            in_channels=self.channels,  out_channels=num_units,
            kernel_size=kernels[idx],  padding=kernels[idx]//2, stride=strides[idx],  dropout=dropouts[idx]
            )
        idx = 1
        self.conv2 = conv_block_spect(
            in_channels=num_units,  out_channels=num_units,
            kernel_size=kernels[idx],  padding=kernels[idx]//2, stride=strides[idx],  dropout=dropouts[idx]
            )
        idx = 2
        self.conv3 = conv_block_spect(
            in_channels=num_units,  out_channels=num_units,
            kernel_size=kernels[idx],  padding=kernels[idx]//2, stride=strides[idx],  dropout=dropouts[idx]
            )
        idx = 3
        self.conv4 = conv_block_spect(
            in_channels=num_units,  out_channels=num_units,
            kernel_size=kernels[idx],  padding=kernels[idx]//2, stride=strides[idx],  dropout=dropouts[idx]
            )
        idx = 4
        self.conv5 = conv_block_spect(
            in_channels=num_units,  out_channels=num_units,
            kernel_size=kernels[idx],  padding=kernels[idx]//2, stride=strides[idx],  dropout=dropouts[idx]
            )
        idx = 5
        self.conv6 = conv_block_spect(
            in_channels=num_units,  out_channels=num_units,
            kernel_size=kernels[idx],  padding=kernels[idx]//2, stride=strides[idx],  dropout=dropouts[idx]
            )
        idx = 5
        self.conv7 = conv_block_spect(
            in_channels=num_units,  out_channels=num_units,
            kernel_size=kernels[idx],  padding=kernels[idx]//2, stride=strides[idx],  dropout=dropouts[idx]
            )
        idx = 7
        self.conv8 = conv_block_spect(
            in_channels=num_units,  out_channels=num_units,
            kernel_size=kernels[idx],  padding=kernels[idx]//2, stride=strides[idx],  dropout=dropouts[idx]
            )
        idx = 8
        self.conv9 = conv_block_spect(
            in_channels=num_units,  out_channels=num_units,
            kernel_size=kernels[idx],  padding=kernels[idx]//2, stride=strides[idx],  dropout=dropouts[idx]
            )
        idx = 9
        self.conv10 = conv_block_spect(
            in_channels=num_units,  out_channels=num_units,
            kernel_size=kernels[idx],  padding=kernels[idx]//2, stride=strides[idx],  dropout=dropouts[idx]
            )
        idx = 10
        self.conv11 = conv_block_spect(
            in_channels=num_units,  out_channels=num_units,
            kernel_size=kernels[idx],  padding=kernels[idx]//2, stride=strides[idx],  dropout=dropouts[idx]
            )
        idx = 11
        self.conv12 = conv_block_spect(
            in_channels=num_units,  out_channels=num_units,
            kernel_size=kernels[idx],  padding=kernels[idx]//2, stride=strides[idx],  dropout=dropouts[idx]
            )
        
        self.conv13 = conv_block_spect(in_channels=num_units,  out_channels=2000, kernel_size=31,  padding=15, stride=1,  dropout=0.3)
        self.conv14 = conv_block_spect(in_channels=2000,  out_channels=2000, kernel_size=1,  padding=0, stride=1,  dropout=0.4)
        self.conv15 = conv_block_spect(in_channels=2000,  out_channels=29, kernel_size=1,  padding=0, stride=1,  dropout=0.0)


    def forward(self, x):
    
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
        out = self.conv15(x)

        out = torch.transpose(out, 1,2)

        return out









################################################################################################
# ####################              Wave2letter_modified         ############################
###############      Was trained till ~8% CER and ~30 %WER
################################################################################################

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
    



### Modified for receptive fields#######

class Wav2LetterRF(pl.LightningModule):
    def __init__(self, model_config = 'config_rf.yaml') -> None:
        super(Wav2LetterRF, self).__init__()
        # Loading the config file..!
        manifest_file = os.path.join(os.path.dirname(__file__), 'conf', model_config)
        with open(manifest_file, 'r') as f:
            self.manifest = yaml.load(f, Loader=yaml.FullLoader)

        self.model_name = "wav2letter_modified"
        self.results_dir = self.manifest["results_dir"]
        self.channels = self.manifest["channels"]
        self.lr = self.manifest["learning_rate"]
        self.processor = Text_Processor()
        self.loss_fn = nn.CTCLoss(blank=28).to(self.device)

        self.validation_step_outputs = []
        self.training_step_outputs = []

        self.conv1 = conv_block(in_channels=self.channels,  out_channels=250, kernel_size=31,  padding=15, stride=20,  dropout=0.2)
        self.conv2 = conv_block(in_channels=250,  out_channels=250, kernel_size=3,  padding=1, stride=2,  dropout=0.2)
        self.conv3 = conv_block(in_channels=250,  out_channels=250, kernel_size=3,  padding=1, stride=2,  dropout=0.2)
        self.conv4 = conv_block(in_channels=250,  out_channels=250, kernel_size=3,  padding=1, stride=2,  dropout=0.2)
        self.conv5 = conv_block(in_channels=250,  out_channels=250, kernel_size=3,  padding=1, stride=2,  dropout=0.2)

        self.conv6 = conv_block(in_channels=250,  out_channels=250, kernel_size=3,  padding=1, stride=1,  dropout=0.2)
        self.conv7 = conv_block(in_channels=250,  out_channels=250, kernel_size=3,  padding=1, stride=1,  dropout=0.2)
        self.conv8 = conv_block(in_channels=250,  out_channels=250, kernel_size=3,  padding=1, stride=1,  dropout=0.2)

        self.conv9 = conv_block(in_channels=250,  out_channels=250, kernel_size=7,  padding=3, stride=1,  dropout=0.2)
        self.conv10 = conv_block(in_channels=250,  out_channels=250, kernel_size=7,  padding=3, stride=1,  dropout= 0.2)
        self.conv11 = conv_block(in_channels=250,  out_channels=250, kernel_size=7,  padding=3, stride=1,  dropout=0.3)
        self.conv12 = conv_block(in_channels=250,  out_channels=250, kernel_size=7,  padding=3, stride=1,  dropout=0.3)
        self.conv13 = conv_block(in_channels=250,  out_channels=2000, kernel_size=31,  padding=15, stride=1,  dropout=0.3)

        self.conv14 = conv_block(in_channels=2000,  out_channels=2000, kernel_size=1,  padding=0, stride=1,  dropout=0.4)
        self.conv15 = conv_block(in_channels=2000,  out_channels=29, kernel_size=1,  padding=0, stride=1,  dropout=0.0)


    def forward(self, x):
        #conv_block ...!
        x = torch.unsqueeze(x, dim=1)
    
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
        out = self.conv15(x)

        out = torch.transpose(out, 1,2)

        return out
    
    def configure_optimizers(self):
        # learning_rate = 1.0e-4
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return optimizer

    def training_step(self, train_batch, batch_idx): 
        
        x,y, input_len, target_len = train_batch

        log_prob = self.forward(x)
        log_prob = nn.functional.log_softmax(log_prob, dim=2).transpose(0,1)  #(time, batch, classes) requried shape for CTCLoss
        loss = self.loss_fn(log_prob, y, input_len, target_len)

        output = {
                'loss': loss,
                # 'log': logs,
                }
        self.training_step_outputs.append(output)
        return output

    def validation_step(self, val_batch, batch_idx):      
        x,y, input_len, target_len = val_batch

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

        output = {'loss': loss, 'cer':cerr, 'wer': werr}
        self.validation_step_outputs.append(output)
        return output
    
    def on_train_epoch_end(self):
        avg_loss = torch.stack([x['loss'] for x in self.training_step_outputs]).mean()
        self.logger.experiment.add_scalar("Loss/Train", avg_loss, self.current_epoch)
        self.training_step_outputs.clear()
        return super().on_train_epoch_end()
    
    def on_validation_epoch_end(self):

        avg_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()
        avg_cer = mean([x['cer'] for x in self.validation_step_outputs])
        avg_wer = mean([x['wer'] for x in self.validation_step_outputs])
        self.logger.experiment.add_scalar("Loss/Val", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("CER/Val", avg_cer, self.current_epoch)
        self.logger.experiment.add_scalar("WER/Val", avg_wer, self.current_epoch)
        
        self.validation_step_outputs.clear()
        return super().on_validation_epoch_end()

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

         
