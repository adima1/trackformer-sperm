import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer

class Transformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()

        # Encoder
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Decoder
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.d_model = d_model
        self.nhead = nhead

    def forward(self, src, mask, query_embed, pos_embed):
        # src: [B, C, H, W]
        # mask: [B, H, W] (bool), True = ignore
        # query_embed: [num_queries, C]
        # pos_embed: [B, C, H, W]

        B, C, H, W = src.shape

        # flatten: [B, C, H*W] -> [H*W, B, C]
        src = src.flatten(2).permute(2, 0, 1)  # [H*W, B, C]
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)  # [H*W, B, C]
        mask = mask.flatten(1)  # [B, H*W]

        # encode
        memory = self.encoder(src + pos_embed, src_key_padding_mask=mask)

        # prepare target input
        query_embed = query_embed.unsqueeze(1).repeat(1, B, 1)  # [num_queries, B, C]
        tgt = torch.zeros_like(query_embed)  # [num_queries, B, C]

        # decode
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask)

        # החזרת hs בצורה תקנית: [num_layers=1, B, num_queries, hidden_dim]
        hs = hs.permute(1, 0, 2).unsqueeze(0)  # [1, B, Q, C]

        return hs, memory.permute(1, 2, 0).view(B, C, H, W)
