
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.distributions.uniform import Uniform

from .position_embedding import add_positional_features

INIT = 1e-2


class Im2LatexModel(nn.Module):
    def __init__(self, out_size, emb_size, dec_rnn_h,
                 enc_out_dim=512, n_layer=1,
                 add_pos_feat=False, dropout=0.):
        super(Im2LatexModel, self).__init__()

        # Vision Transformer encoder
        self.vit_encoder = vit_b_16(pretrained=True)
        self.vit_encoder.heads = nn.Identity()  # Remove classification head

        # Transformer decoder
        config = GPT2Config(vocab_size=out_size, n_embd=emb_size, n_layer=n_layer, n_head=8)
        self.transformer_decoder = GPT2LMHeadModel(config)
        self.embedding = nn.Embedding(out_size, emb_size)

        self.init_wh = nn.Linear(enc_out_dim, dec_rnn_h)
        self.init_wc = nn.Linear(enc_out_dim, dec_rnn_h)
        self.init_wo = nn.Linear(enc_out_dim, dec_rnn_h)

        self.add_pos_feat = add_pos_feat
        self.dropout = nn.Dropout(p=dropout)
        self.uniform = Uniform(0, 1)

    def forward(self, imgs, formulas, epsilon=1.):
        # Encoding
        encoded_imgs = self.encode(imgs)  # [B, 197, 768]

        # Prepare the transformer decoder input
        max_len = formulas.size(1)
        tgt_embeddings = self.embedding(formulas)  # [B, MAX_LEN, emb_size]

        # Generate sequence
        logits = self.transformer_decoder(inputs_embeds=tgt_embeddings, encoder_hidden_states=encoded_imgs).logits
        return logits

    def encode(self, imgs):
        encoded_imgs = self.vit_encoder(imgs)  # [B, 197, 768]
        if self.add_pos_feat:
            encoded_imgs = add_positional_features(encoded_imgs)
        return encoded_imgs

    def init_decoder(self, enc_out):
        mean_enc_out = enc_out.mean(dim=1)
        h = self._init_h(mean_enc_out)
        c = self._init_c(mean_enc_out)
        init_o = self._init_o(mean_enc_out)
        return (h, c), init_o

    def _init_h(self, mean_enc_out):
        return torch.tanh(self.init_wh(mean_enc_out))

    def _init_c(self, mean_enc_out):
        return torch.tanh(self.init_wc(mean_enc_out))

    def _init_o(self, mean_enc_out):
        return torch.tanh(self.init_wo(mean_enc_out))
