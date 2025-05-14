import os


# local
from wav2letter.utils.cer import cer
from wav2letter.utils.wer import wer
from wav2letter.config.evaluation_config import EvalConfig

from wav2letter.datasets import DataModuleRF, LibriSpeechDataset 
from wav2letter.models import Wav2LetterRF
from wav2letter import manifest


def evaluate(cfg: EvalConfig):
    """Computes CER and WER for the given checkpoint, on the test set."""
    res_dir = manifest['results_dir']
    checkpoint_path = os.path.join(res_dir, cfg.checkpoint)
    model = Wav2LetterRF.load_from_checkpoint(checkpoint_path=checkpoint_path)

    dataset = DataModuleRF()
    dataset.adv_robust = cfg.adv_robust
    dataset.setup()
    test_dataloader = dataset.val_dataloader()
    total_CER = []
    total_WER = []

    for idx, test_batch in enumerate(test_dataloader):
        x,y, input_len, target_len = test_batch
        # print(f"batch_size: {x.shape[0]}")

        # log_prob = model.forward(x)
        # log_prob = nn.functional.log_softmax(log_prob, dim=2).transpose(0,1)  #(time, batch, classes) requried shape for CTCLoss
        # loss = self.loss_fn(log_prob, y, input_len, target_len)
        # self.log('val_loss', loss, on_step=False, on_epoch=True)
        pred = model.decode(x)
        target = model.processor.label_to_text(y)
        cer_cum = []
        wer_cum = []
        for k in range(len(pred)):
            cer_cum.append(cer(target[k], pred[k]))
            wer_cum.append(wer(target[k], pred[k]))
        batch_cerr = sum(cer_cum)/len(cer_cum)
        batch_werr = sum(wer_cum)/len(wer_cum)
        total_CER.append(batch_cerr)
        total_WER.append(batch_werr)
    CER = sum(total_CER)/len(total_CER)
    WER = sum(total_WER)/len(total_WER)
    output = {'cer': CER, 'wer': WER}
    return output

    
if __name__ == "__main__":
    cfg = EvalConfig()

    output = evaluate(cfg)
    print(f"For the checkpoint: '{cfg.checkpoint}', \
        evaluation results are as follows:\n \
        CER: {output['cer']:.2f}, \n WER: {output['wer']:.2f}.")
    


