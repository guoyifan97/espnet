# The implementation of DPRNN in
# Luo. et al. "Dual-path rnn: efficient long sequence modeling
# for time-domain single-channel speech separation."
#
# The code is based on:
# https://github.com/yluo42/TAC/blob/master/utility/models.py
#


import torch
from torch.autograd import Variable
import torch.nn as nn

from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.encoder import (
    Encoder as TransformerEncoder,
    repeat,
)
from espnet.nets.pytorch_backend.conformer.encoder import (
    Encoder as ConformerEncoder,  # noqa: H301
)
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask

EPS = torch.finfo(torch.get_default_dtype()).eps

# dual-path RNN
class SepFormer(nn.Module):
    """SepFormer.

    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        output_size: int, dimension of the output size.
        dropout: float, dropout ratio. Default is 0.
        num_layers: int, number of stacked RNN layers. Default is 1.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is True.
    """

    def __init__(
        self,
        input_size,
        num_spk,
        encoder_type: str = "transformer",
        num_blocks: int = 2,
        N_intra: int = 8,
        N_inter: int = 8,
        adim: int = 256,
        aheads: int = 8,
        dropout_rate: float = 0.0,
        linear_units=2048,
        positional_dropout_rate=0.1,
        attention_dropout_rate=0.0,
        input_layer=None,
        pos_enc_class=PositionalEncoding,
        normalize_before=True,
        concat_after=False,
        positionwise_layer_type="linear",
        positionwise_conv_kernel_size=1,
        selfattention_layer_type="selfattn",
        stochastic_depth_rate=0.0,
        intermediate_layers=None,
        nonlinear="prelu",
    ):
        super().__init__()

        assert encoder_type in ["transformer", "conformer"], f"{encoder_type}"
        if encoder_type == "conformer":
            raise NotImplementedError()

        output_size = input_size * num_spk

        # SepTransformer
        self.sepformer = nn.ModuleList()

        for i in range(num_blocks):
            sepformer_block = nn.ModuleList()
            sepformer_block.append(
                TransformerEncoder(
                    idim=input_size if i == 0 else adim,
                    num_blocks=N_intra,
                    attention_dim=adim,
                    attention_heads=aheads,
                    linear_units=linear_units,
                    positional_dropout_rate=positional_dropout_rate,
                    attention_dropout_rate=attention_dropout_rate,
                    dropout_rate=dropout_rate,
                    input_layer="linear" if i == 0 else None,
                    pos_enc_class=pos_enc_class,
                    normalize_before=normalize_before,
                    concat_after=concat_after,
                    positionwise_layer_type=positionwise_layer_type,
                    positionwise_conv_kernel_size=positionwise_conv_kernel_size,
                    selfattention_layer_type=selfattention_layer_type,
                    stochastic_depth_rate=stochastic_depth_rate,
                    intermediate_layers=intermediate_layers,
                )
            )
            sepformer_block.append(
                TransformerEncoder(
                    idim=adim,
                    num_blocks=N_inter,
                    attention_dim=adim,
                    attention_heads=aheads,
                    linear_units=linear_units,
                    positional_dropout_rate=positional_dropout_rate,
                    attention_dropout_rate=attention_dropout_rate,
                    dropout_rate=dropout_rate,
                    input_layer=input_layer,
                    pos_enc_class=pos_enc_class,
                    normalize_before=normalize_before,
                    concat_after=concat_after,
                    positionwise_layer_type=positionwise_layer_type,
                    positionwise_conv_kernel_size=positionwise_conv_kernel_size,
                    selfattention_layer_type=selfattention_layer_type,
                    stochastic_depth_rate=stochastic_depth_rate,
                    intermediate_layers=intermediate_layers,
                )
            )

            self.sepformer.append(sepformer_block)

        # output layer
        if nonlinear not in ("prelu","sigmoid", "relu", "tanh"):
            raise ValueError("Not supporting nonlinear={}".format(nonlinear))
        
        self.nonlinear = {
            "prelu": torch.nn.PReLU(),
            "sigmoid": torch.nn.Sigmoid(),
            "relu": torch.nn.ReLU(),
            "tanh": torch.nn.Tanh(),
        }[nonlinear]
        self.output = nn.Sequential(nn.PReLU(), nn.Conv2d(adim, output_size, 1))

    def forward(self, xs, ilens):
        # input shape: batch, N, dim1, dim2
        # dim1 = C, dim2 = T
        # apply RNN on dim1 first and then dim2
        # output shape: B, output_size, dim1, dim2
        # input = input.to(device)

        batch_size, _, C, T = xs.shape
        output = xs
        ilens = ilens // C

        for i in range(len(self.sepformer)):

            intra_input = output.permute(0, 3, 2, 1) \
                .contiguous() \
                .view(batch_size * T, C, -1)
            # all one
            intra_mask = torch.ones(batch_size * T, 1, C).to(intra_input.device)
            intra_output, _ = self.sepformer[i][0](intra_input, intra_mask)
            intra_output = intra_output \
                .view(batch_size, T, C, -1) \
                .permute(0, 3, 2, 1) \
                .contiguous() # B F C T
            output = intra_output

            inter_input = output.permute(0, 2, 3, 1) \
                .contiguous() \
                .view(batch_size * C, T, -1) 
            inter_mask = make_non_pad_mask(ilens, torch.zeros(batch_size, C, T)) \
                .contiguous() \
                .view(batch_size * C, 1, -1) \
                .to(inter_input.device)

            inter_output, _ = self.sepformer[i][1](inter_input, inter_mask)
            inter_output = inter_output.view(batch_size, C, T, -1) \
                .permute(0, 3, 1, 2) \
                .contiguous() # B F C T
            output = inter_output

        output = self.nonlinear(output)
        output = self.output(output)  # B, output_size, dim1, dim2

        return output

if __name__ == "__main__":
    m = SepFormer(128, 2)
    print(m)
    xs = torch.rand(3, 128, 5, 1000)
    ilens = torch.tensor([700,900,1000])
    m(xs, ilens)