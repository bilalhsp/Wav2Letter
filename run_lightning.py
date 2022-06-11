import os
print(os.environ.get('CUDA_VISIBLE_DEVICES', []))
import torch
print(torch.cuda.device_count())
print(torch.cuda.get_device_name())
from wav2letter.ASR.dataloader.dataset import Dataset, LSDataModule
from wav2letter.ASR.model import LitWav2Letter
import yaml
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
#from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning import loggers as pl_loggers

#import time
import torchaudio
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(model_param):
    #start_time = time.time()


    # Loading the config. file...!

    #Model and other classes...!
    #torchaudio.datasets.LIBRISPEECH(model_param['data_dir'], url="train-other-500", download = True)
    #train_data = Dataset(model_param)
    #test_data = Dataset(model_param,test=True)
    
    # train_loader = train_data.load_data(batch_size=hyper_param['batch_size'])
    # test_loader = test_data.load_data(batch_size=hyper_param['batch_size'])
    
    print("Creating dataset...!")
    dm = LSDataModule(model_param)
    print("Dataset ready...!")
    mod = LitWav2Letter(model_param)
    # Explicitly specify the process group backend if you choose to
    logs_dir = os.path.join(model_param["results_dir"], "logs")
    #ddp = DDPStrategy(process_group_backend="nccl")
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=logs_dir)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=model_param["results_dir"],
        filename="Wav2Letter-{epoch:03d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
        every_n_epochs=5)
    trainer = pl.Trainer(gpus=1,#[0,1], num_nodes=4,
            # auto_select_gpus=True, 
            # max_epochs=1,
            max_steps=10, 
            logger = tb_logger,
            # profiler="pytorch",

            callbacks=[checkpoint_callback],
            #enable_checkpointing=False,
            progress_bar_refresh_rate=200,
            #accumulate_grad_batches=4,
            accelerator='gpu',#strategy='ddp',
            #devices=2,
            limit_train_batches=0.001, limit_val_batches=0.1,
            #log_gpu_memory=True,
            #deterministic=True,
            )
    print("Training starts now...!")
    trainer.fit(mod, dm)#, ckpt_path="/scratch/gilbreth/ahmedb/wav2letter/lightning/Wav2Letter-epoch=074-val_loss=0.56.ckpt",)



####### Training ended...############
    # end_time = time.time()
    # exe_time = (end_time-start_time)/60
    # print(f"It took {exe_time:.3f} min. to train for {hyper_param['epochs']} epochs")
    print("Training on lightening ended succesfully...!")

if __name__ == '__main__':
    pl.utilities.seed.seed_everything(1)
    dir = os.getcwd()
    conf_file = 'lightning.yaml'
    #conf_file = 'data360.yaml'
    manifest_file = os.path.join(dir,"conf",conf_file)
    with open(manifest_file, 'r') as f:
        manifest = yaml.load(f, Loader=yaml.FullLoader)
    model_param = manifest#['model_param']
    # hyper_param = manifest['hyper_param']
    print("Config. file loaded..!")
    print(f"Batch size: {model_param['batch_size']}")
    main(model_param)