import torch
import pandas as pd
import numpy as np
import math
import torch.nn.functional as F
import torch.nn as nn


class InputEmbedding(nn.Module):
  def __init__(self, d_model:int, vocabs:int):
    super().__init__()
    #Dimension of the token's vectors
    self.d_model = d_model
    #Token dictionary
    self.vocabs=vocabs
    #Embedding layer
    self.embedding = nn.Embedding(self.vocabs, self.d_model)

  def forward(self, x):
    '''
    args:
        x: input of shape (batch size,sequence length)
    returns:
        output of (batch size, sequence length, embedding vectors)
    '''
    return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
  def __init__(self, Seq_len:int, d_model:int, dropout:float) -> None:
    super().__init__()
    self.Seq_len = Seq_len
    self.d_model = d_model
    self.dropout = nn.Dropout(dropout)

    pe = torch.zeros(Seq_len, d_model)
    position = torch.arange(0, Seq_len, dtype=torch.float).unsqueeze(1)
    pos_denom = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * pos_denom)
    pe[:,1::2] =  torch.cos(position * pos_denom)
    pe = pe.unsqueeze(0)
    self.register_buffer("pe", pe) #buffers- tensors not considered as model parameters


  def forward(self, x):
    '''
    Args:
        x: Input of size(batch, sequence length, embedding dimension)
    output:
          output is the sum of Inputs and Positional encodings values.
    '''
    # (batch, Seq_len, d_model)
    x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
    #to prevent model on over-relying on positions, some embedding values are zeroed out during training.
    return self.dropout(x)


class MultiHeadAttentionBlock(nn.Module):

  def __init__(self, d_model: int, h: int, dropout: float) -> None:
      super().__init__()
      self.d_model = d_model
      self.h = h # Number of heads
      # Make sure d_model is divisible by h
      assert d_model % h == 0, "Vector Dimension is not divisible by h"

      self.d_k = d_model // h # Dimension of vector seen by each head
      self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq
      self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
      self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv
      self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo
      self.dropout = nn.Dropout(dropout)

  @staticmethod
  def attention(query, key, value, mask, dropout: nn.Dropout):
      d_k = query.shape[-1]
      # Just apply the formula from the paper
      # (batch, h, Seq_len, d_k) --> (batch, h, Seq_len, Seq_len)
      attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
      if mask is not None:
          # Write a very low value (indicating -inf) to the positions where mask == 0
          attention_scores.masked_fill_(mask == 0, -1e9)
      attention_scores = attention_scores.softmax(dim=-1)  # Apply softmax
      if dropout is not None:
          attention_scores = dropout(attention_scores)
      # (batch, h, Seq_len, Seq_len) --> (batch, h, Seq_len, d_k)
      # return attention scores which can be used for visualization
      return (attention_scores @ value), attention_scores

  def forward(self, q, k, v, mask):
      query = self.w_q(q) # (batch, Seq_len, d_model) * (d_model, d_model)
      key = self.w_k(k)
      value = self.w_v(v)

      # (batch, Seq_len, d_model) --> (batch, Seq_len, h, d_k) --> (batch, h, Seq_len, d_k)
      query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
      key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
      value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

      # Calculate attention
      x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

      # Combine all the heads together
      # (batch, h, Seq_len, d_k) --> (batch, Seq_len, h, d_k) --> (batch, Seq_len, d_model)
      x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

      # Multiply by Wo
      # (batch, Seq_len, d_model) --> (batch, Seq_len, d_model)
      return self.w_o(x)

class LayerNormalization(nn.Module):
  def __init__(self, eps:float = 10**-6) -> None:
    super().__init__()
    self.eps = eps
    self.alpha = nn.Parameter(torch.ones(1))
    self.bias = nn.Parameter(torch.zeros(1))

  def forward(self, x):
    mean = x.mean(dim =-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    return self.alpha * (x - mean) / (std + self.eps) + self.bias


class ResidualConnection(nn.Module):

  def __init__(self, features: int, dropout: float) -> None:
      super().__init__()
      self.dropout = nn.Dropout(dropout)
      self.norm = LayerNormalization(features)

  def forward(self, x, sublayer):
      #From the paper, we do post-normalization against pre-norm
      return x + self.dropout(self.norm(sublayer(x)))


class FeedForwardNet(nn.Module):

  def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
      super().__init__()
      self.linear_1 = nn.Linear(d_model, d_ff)
      self.dropout = nn.Dropout(dropout)
      self.linear_2 = nn.Linear(d_ff, d_model)

  def forward(self, x):
      # (batch, Seq_len, d_model) --> (batch, Seq_len, d_ff) --> (batch, Seq_len, d_model)
      return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class ProjectionLayer(nn.Module): #this is the linear layer

  def __init__(self, d_model, vocab_size) -> None:
      super().__init__()
      self.proj = nn.Linear(d_model, vocab_size)

  def forward(self, x) -> None:
      # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
      return torch.log_softmax(self.proj(x), dim=-1)

class EncoderBlock(nn.Module):

  def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_Net: FeedForwardNet, dropout: float) -> None:
      super().__init__()
      self.self_attention_block = self_attention_block
      self.feed_forward_Net = feed_forward_Net
      self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)]) # Module list for 2 skip connections.

  def forward(self, x, enc_mask):  #encoder_mask to hide "[pad]" term interaction's with other words.
      x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, enc_mask))
      x = self.residual_connections[1](x, self.feed_forward_Net)
      return x

class Encoder(nn.Module):

  def __init__(self, features: int, layers: nn.ModuleList) -> None:
      super().__init__()
      self.layers = layers
      self.norm = LayerNormalization(features)

  def forward(self, x, mask):
      for layer in self.layers:
          x = layer(x, mask)
      return self.norm(x)

class DecoderBlock(nn.Module):

  def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_Net: FeedForwardNet, dropout: float) -> None:
      super().__init__()
      self.self_attention_block = self_attention_block
      self.cross_attention_block = cross_attention_block
      self.feed_forward_Net = feed_forward_Net
      self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

  def forward(self, x, encoder_output, enc_mask, dec_mask):
      x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, dec_mask))
      x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, enc_mask))
      x = self.residual_connections[2](x, self.feed_forward_Net)
      return x

class Decoder(nn.Module):

  def __init__(self, features: int, layers: nn.ModuleList) -> None:
      super().__init__()
      self.layers = layers
      self.norm = LayerNormalization(features)

  def forward(self, x, encoder_output, enc_mask, dec_mask):
      for layer in self.layers:
          x = layer(x, encoder_output, enc_mask, dec_mask)
      return self.norm(x)


class Transformer(nn.Module):

  def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding, tgt_embed: InputEmbedding, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
      super().__init__()
      self.encoder = encoder
      self.decoder = decoder
      self.src_embed = src_embed #inputEmbedding
      self.tgt_embed = tgt_embed #inputEmbedding
      self.src_pos = src_pos   #positionalencoding
      self.tgt_pos = tgt_pos   #positionalencoding
      self.projection_layer = projection_layer

  def encode(self, src, enc_mask):
      # (batch, seq_len, d_model)
      src = self.src_embed(src)
      src = self.src_pos(src)
      return self.encoder(src, enc_mask)

  def decode(self, encoder_output: torch.Tensor, enc_mask: torch.Tensor, tgt: torch.Tensor, dec_mask: torch.Tensor):
      # (batch, seq_len, d_model)
      tgt = self.tgt_embed(tgt)
      tgt = self.tgt_pos(tgt)
      return self.decoder(tgt, encoder_output, enc_mask, dec_mask)

  def project(self, x):
      # (batch, seq_len, vocab_size)
      return self.projection_layer(x)


def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=2, h: int=4, dropout: float=0.1, d_ff: int=2048) -> Transformer:
    # Create the embedding layers
    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(src_seq_len, d_model, dropout) #~fix this
    tgt_pos = PositionalEncoding(tgt_seq_len, d_model, dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardNet(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardNet(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
