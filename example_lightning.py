import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

import pytorch_lightning as ptl
from torch.utils.data import random_split
from pytorch_lightning import Trainer

class CoolModel(ptl.LightningModule):

    def __init__(self):
        super(CoolModel, self).__init__()
        # not the best model...
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def my_loss(self, y_hat, y):
        return F.cross_entropy(y_hat, y)

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        return {'loss': self.my_loss(y_hat, y)}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        return {'val_loss': self.my_loss(y_hat, y)}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'avg_val_loss': avg_loss}

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=0.02)]

    
    def train_dataloader(self):
        return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=32)

    
    def val_dataloader(self):
        return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=32)

    
    def test_dataloader(self):
        return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=32)







def main():
    # data
    # dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
    # mnist_train, mnist_val = random_split(dataset, [55000, 5000])
    # train_loader = DataLoader(mnist_train, batch_size=32)
    # val_loader = DataLoader(mnist_val, batch_size=32)
    model = CoolModel()
    #exp = Experiment(save_dir=os.getcwd())
    
    # train on 80 GPUs across 10 nodes
    trainer = Trainer(accelerator='ddp', 
                      #max_epochs=1, 
                      gpus=[0], 
                      num_nodes=1,
                      
                      )
    trainer.fit(model)#, train_loader, val_loader)
  
if __name__ ==  '__main__':
    main()