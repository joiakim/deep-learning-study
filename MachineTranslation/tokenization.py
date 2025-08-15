from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
from datasets import load_dataset
from pathlib import Path

class train_tokenizer:
  def __init__(self, dataset, text_column, lang):
    self.dataset = dataset
    self.text_column = text_column
    self.lang = lang

  @staticmethod
  def get_training_corpus(self):
    for i in range(0, len(self.dataset),1000):
      yield self.dataset[i :i+1000][self.text_column]

  def WordpieceTokenizer(self):
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence(
      [normalizers.Lowercase()]
    )
    pre_tokenizer = pre_tokenizers.Sequence(
      [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()]
    )
    special_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]", "[MASK]"]
    trainer = trainers.WordPieceTrainer(vocab_size=11500, special_tokens=special_tokens, min_frequency=2)
    tokenizer.train_from_iterator(train_tokenizer.get_training_corpus(self), trainer=trainer)
    tokenizer.save(f'tokenizer_{self.lang}.json')
    return tokenizer

#   def BPEtokenizer(self):
#     tokenizerbpe = Tokenizer(models.BPE())
#     tokenizerbpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
#     trainerbpe = trainers.BpeTrainer(vocab_size=11500, special_tokens=["<|startoftext|>","<|endoftext|>"])
#     tokenizerbpe.train_from_iterator(train_tokenizer.get_training_corpus(self), trainer=trainerbpe)
#     tokenizerbpe.post_processor = processors.ByteLevel(trim_offsets=False)
#     return tokenizerbpe

#   def wordleveltokenizer(self):
#     special_tokens = ["<UNK>", "<SOS>", "<SEP>", "<PAD>", "<MASK>"]
#     tokenizerwl = Tokenizer(models.WordLevel(unk_token="<UNK>"))
#     tokenizerwl.pre_tokenizer = pre_tokenizers.Whitespace()
#     trainerwl = trainers.WordLevelTrainer(vocab_size=2000,special_tokens=special_tokens, min_frequency=2)
#     tokenizerwl.train_from_iterator(train_tokenizer.get_training_corpus(self), trainerwl)
#     return tokenizerwl

# 36k twi sentences and 39k english sentences to train our tokenizer for inferencing during training

def tokenizer_choice(tkz):
  Eng_data = load_dataset("Harsit/xnli2.0_train_english")['train']
  Twi_data = load_dataset("michsethowusu/twi_sentences")['train']
  tokenize_eng = train_tokenizer(Eng_data, 'premise', 'eng')
  tokenize_twi = train_tokenizer(Twi_data, 'sentence', 'twi')
  tokenizer_eng = getattr(tokenize_eng, tkz)
  tokenizer_twi = getattr(tokenize_twi, tkz)
  print("building Tokenizers")
  return tokenizer_twi(), tokenizer_eng()