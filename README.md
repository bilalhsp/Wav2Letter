# 🗣️ Wav2LetterRF — Convolutional ASR Model in PyTorch Lightning

This repository provides an implementation of a modified **Wav2Letter** architecture (`Wav2LetterRF`) for Automatic Speech Recognition (ASR), built using **PyTorch Lightning**. The model is trained using **Connectionist Temporal Classification (CTC)** loss and is trained with the **LibriSpeech** dataset.

---

## 📦 Features

- 🧱 Fully convolutional architecture for ASR
- 🧪 CTCLoss for alignment-free training
- 📚 Built-in support for LibriSpeech dataset
- 🔁 Logging of training/validation loss, WER, and CER
- 📤 Easy decoding and inference pipeline
- ✅ Ready-to-use pretrained model interface

---

## 🛠️ Installation

```bash
git clone https://github.com/bilalhsp/Wav2Letter.git
cd Wav2Letter
pip install -e .
```

---

## 📁 Project Structure

```
├── scripts/                    # Training and inference scripts
└── wav2letter/
    │
    ├── datasets.py             # pytorch dataset class for LibriSpeech 
    ├── models.py               # Contains Wav2LetterRF model
    ├── prepareLibriSpeech.py   # Helper functions for downloadin LibriSpeech         
    ├── conf/                   # YAML config files
    ├── utils/                  # Helper functions (decoding, evaluation)
    └── ...
```

---

## 📚 LibriSpeech Support

The repository includes code to:

- ✅ Download LibriSpeech splits
- ✅ Preprocess and prepare dataloaders
- ✅ Tokenize and decode transcripts

---

## 🚀 Usage Example

### ✨ Load Pretrained Model and Transcribe

```python
from wav2letter.models import Wav2LetterRF
import torch
import torchaudio

# Load pretrained model (update path if needed)
checkpoint = 'path_to_pretrained_weights'
model = Wav2LetterRF.load_from_checkpoint(checkpoint)
model.eval()

# Load and preprocess an audio file
waveform, sample_rate = torchaudio.load("sample.wav")
if sample_rate != 16000:
    waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
waveform = waveform.squeeze(0)  # Mono

# Transcribe
with torch.no_grad():
    pred_text = model.decode(waveform.unsqueeze(0))
print("Predicted transcription:", pred_text[0])
```

---

## ⚙️ Training

To train the model:

```bash
python scripts/run_lightning.py
```

You can customize the YAML config for:

- Learning rate
- Number of channels
- Dataset paths
- Output directory

---

## 📊 Logging & Evaluation

Validation loss, WER (Word Error Rate), and CER (Character Error Rate) are automatically logged using your preferred logger (e.g., TensorBoard).

---

## 📝 References

- Collobert et al., "Wav2Letter: an End-to-End ConvNet-based Speech Recognition System"
- [LibriSpeech ASR corpus](http://www.openslr.org/12/)

---

## 📩 Contact

For questions or issues, feel free to open an issue or reach out.

