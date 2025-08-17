# Transformer Translation Model

A ongoing implementation of a Transformer model for neural machine translation for English to Twi and Vice Versa.

## 🌟 Features

- **Complete Transformer Architecture**: Multi-head attention, positional encoding, layer normalization.
- **Static Dataset Support**: Built for English to twi and twi to English.
- **Scripts**: Includes training and evaluation.
- **Advanced Decoding**: greedy decoding 


## 🔻 shortfalls

-  Poor and paltry Data for both tokenizer and inference.
-  Poor source and target language pairs.

## 🛠️ Upgrades
-  curate High quality data for Translation task.
-  Add inference files, tokenizer files and etc.
-  Make model more configurable.
-  Upload Model to Huggingface. 
  
## 🏗️ Architecture

The model implements the "Attention Is All You Need" paper architecture:

- **Encoder-Decoder Structure**: 2 encoder and 2 decoder layers (configurable)
- **Multi-Head Attention**: 4 attention heads with 612-dimensional model
- **Feed-Forward Networks**: 2048-dimensional inner layer
- **Positional Encoding**: Sinusoidal positional embeddings
- **Layer Normalization**: Applied to all sub-layers
- **Residual Connections**: Around all sub-layers

## 📁 Project Structure

```
MachineTranslation/
├── model.py              # Core Transformer implementation
├── dataset.py            # Dataset loading and preprocessing
├── train.py              # Training script with validation
├── README.md             # This file
├── tokenization.py       # tokenizer training 
    

```

## 🚀 Quick Start

### 1. Setup and Installation

```bash
# Clone or create the project directory
mkdir MachineTranslation
cd MachineTranslation

# Run  setup (installs dependencies)
pip install -r requirements.txt
```

### 2. Train the Model

```bash
# Train with default configuration (Twi to English)
python setup.py --train

# Or run training directly
python train.py
```


## ⚙️ Configuration


```json
{
  "batch_size": 8,
  "num_epochs": 20,
  "lr": 1e-3,
  "seq_len": 500,
  "d_model": 612,
  "datasource": "Huggingface/michsethowusu",
  "lang_src": "twi",
  "lang_tgt": "Eng",

}
```

###
