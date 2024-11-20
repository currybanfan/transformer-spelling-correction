from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from torch import Tensor
from torch import nn
import torch
import math
from torch.nn import functional as F



class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, batch_first: bool = False):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        if self.batch_first:
            x = x.transpose(0, 1)
            x = x + self.pe[:x.size(0)]
            return self.dropout(x.transpose(0, 1))
        else:
            x = x + self.pe[:x.size(0)]
            return self.dropout(x)
        
class Encoder(nn.Module):
    def __init__(self, num_emb, hid_dim, n_layers, n_heads, ff_dim, dropout, max_length=100):
        super(Encoder, self).__init__()
        self.tok_embedding = nn.Embedding(num_emb, hid_dim)
        self.pos_embedding = PositionalEncoding(hid_dim, dropout, max_length, batch_first=True)
        self.layer = TransformerEncoderLayer(d_model=hid_dim, nhead=n_heads, dim_feedforward=ff_dim, dropout=dropout, batch_first=True)
        self.encoder = TransformerEncoder(self.layer, num_layers=n_layers)

    def forward(self, src, src_mask=None, src_pad_mask=None):
        # 將輸入通過嵌入層和位置編碼層，然後傳入編碼器
        src = self.tok_embedding(src)
        src = self.pos_embedding(src)
        src = self.encoder(src, mask=src_mask, src_key_padding_mask=src_pad_mask)
        return src

class Decoder(nn.Module):
    def __init__(self, num_emb, hid_dim, n_layers, n_heads, ff_dim, dropout, max_length=100):
        super(Decoder, self).__init__()
        self.tok_embedding = nn.Embedding(num_emb, hid_dim)
        self.pos_embedding = PositionalEncoding(hid_dim, dropout, max_length, batch_first=True)
        self.layer = TransformerDecoderLayer(d_model=hid_dim, nhead=n_heads, dim_feedforward=ff_dim, dropout=dropout, batch_first=True)
        self.decoder = TransformerDecoder(self.layer, num_layers=n_layers)

    def forward(self, tgt, memory, tgt_mask=None, tgt_pad_mask=None, memory_key_padding_mask=None):
        # 將目標通過嵌入層和位置編碼層，然後傳入解碼器
        tgt = self.tok_embedding(tgt)
        tgt = self.pos_embedding(tgt)
        tgt = self.decoder(tgt, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_pad_mask, memory_key_padding_mask=memory_key_padding_mask)
        return tgt

class TransformerAutoEncoder(nn.Module):
    def __init__(self, num_emb, hid_dim, n_layers, n_heads, ff_dim, dropout, max_length):
        super(TransformerAutoEncoder, self).__init__()
        self.encoder = Encoder(num_emb, hid_dim, n_layers, n_heads, ff_dim, dropout, max_length)
        self.decoder = Decoder(num_emb, hid_dim, n_layers, n_heads, ff_dim, dropout, max_length)
        self.fc_out = nn.Linear(hid_dim, num_emb) 
        self.register_buffer('tgt_mask', self.gen_mask(max_length))


    def gen_mask(self, size):
        # triu mask for decoder
        mask = torch.triu(torch.ones((size, size)), diagonal=1).bool()
        return mask

    def forward(self, src, tgt, src_pad_mask, tgt_pad_mask):
        """
        src_pad_mask : 對 src 的填充位置進行遮蔽
        tgt_mask : 目標序列的當前位置只能「看到」其之前的 token
        tgt_pad_mask : 對 tgt 的填充位置進行遮蔽
        memory_key_padding_mask : 用於遮蔽編碼器輸出的填充位置
        """
        enc_src = self.encoder(src, src_pad_mask=src_pad_mask)
        out = self.decoder(tgt, enc_src, tgt_mask=self.tgt_mask, tgt_pad_mask=tgt_pad_mask, memory_key_padding_mask=src_pad_mask)
        out = self.fc_out(out)
        return out
    

