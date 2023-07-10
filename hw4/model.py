import torch.nn as nn

from conformer import ConformerBlock
from self_attention_pooling import SelfAttentionPooling


class Classifier(nn.Module):
    def __init__(self, d_model=256, n_spks=600, dropout=0.1):
        super().__init__()
        # Project the dimension of features from that of input into d_model.
        self.prenet = nn.Linear(40, d_model)
        # TODO:
        #   Change Transformer to Conformer.
        #   https://arxiv.org/abs/2005.08100
        # self.encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=d_model, dim_feedforward=256, nhead=1
        # )

        self.encoder_layer = ConformerBlock(dim=256)
        # self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

        self.sap = SelfAttentionPooling(input_dim=256)
        # Project the dimension of features from d_model into speaker nums.
        self.pred_layer = nn.Sequential(
            nn.Linear(d_model, n_spks),
        )

    def forward(self, mels):
        """
        args:
            mels: (batch size, length, 40)
        return:
            out: (batch size, n_spks)
        """
        # out: (batch size, length, d_model)
        out = self.prenet(mels)
        # out: (length, batch size, d_model)
        out = out.permute(1, 0, 2)
        # The encoder layer expect features in the shape of (length, batch size, d_model).
        out = self.encoder_layer(out)
        # out: (batch size, length, d_model)
        out = out.transpose(0, 1)
        # mean pooling
        # stats = out.mean(dim=1)
        stats = self.sap(out)

        # out: (batch, n_spks)
        out = self.pred_layer(stats)
        return out
