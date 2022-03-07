from collections import OrderedDict
from distutils.version import LooseVersion
from typing import List
from typing import Tuple
from typing import Union

import torch
import random
from torch_complex.tensor import ComplexTensor

from espnet2.enh.layers.complex_utils import is_complex
from espnet2.enh.layers.tcn_tac import TemporalConvNetTAC
from espnet2.enh.separator.abs_separator import AbsSeparator


is_torch_1_9_plus = LooseVersion(torch.__version__) >= LooseVersion("1.9.0")


class TCNTACSeparator(AbsSeparator):
    def __init__(
        self,
        input_dim: int,
        num_spk: int = 2,
        layer: int = 8,
        stack: int = 3,
        bottleneck_dim: int = 128,
        hidden_dim: int = 512,
        kernel: int = 3,
        causal: bool = False,
        norm_type: str = "gLN",
        nonlinear: str = "relu",
        training_channel_num: int = -1,
    ):
        """Temporal Convolution Separator

        Args:
            input_dim: input feature dimension
            num_spk: number of speakers
            layer: int, number of layers in each stack.
            stack: int, number of stacks
            bottleneck_dim: bottleneck dimension
            hidden_dim: number of convolution channel
            kernel: int, kernel size.
            causal: bool, defalut False.
            norm_type: str, choose from 'BN', 'gLN', 'cLN'
            nonlinear: the nonlinear function for mask estimation,
                       select from 'relu', 'tanh', 'sigmoid'
            training_channel_num: the num of channels used to train
        """
        super().__init__()

        self._num_spk = num_spk

        if nonlinear not in ("sigmoid", "relu", "tanh"):
            raise ValueError("Not supporting nonlinear={}".format(nonlinear))

        self.tcn_tac = TemporalConvNetTAC(
            N=input_dim,
            B=bottleneck_dim,
            H=hidden_dim,
            P=kernel,
            X=layer,
            R=stack,
            C=num_spk,
            norm_type=norm_type,
            causal=causal,
            mask_nonlinear=nonlinear,
        )

        self.channel_num = training_channel_num

    def forward(
        self, input: Union[torch.Tensor, ComplexTensor], ilens: torch.Tensor
    ) -> Tuple[List[Union[torch.Tensor, ComplexTensor]], torch.Tensor, OrderedDict]:
        """Forward.

        Args:
            input (torch.Tensor or ComplexTensor): Encoded feature [B, T, N, C]
            ilens (torch.Tensor): input lengths [Batch]

        Returns:
            masked (List[Union(torch.Tensor, ComplexTensor)]): [(B, T, N), ...]
            ilens (torch.Tensor): (B,)
            others predicted data, e.g. masks: OrderedDict[
                'mask_spk1': torch.Tensor(Batch, Frames, Freq, Channel),
                'mask_spk2': torch.Tensor(Batch, Frames, Freq, Channel),
                ...
                'mask_spkn': torch.Tensor(Batch, Frames, Freq, Channel),
            ]
        """
        assert input.dim() == 4

        # if complex spectrum
        if is_complex(input):
            feature = abs(input)
        else:
            feature = input

        B, T, N, C = feature.shape

        if self.training and self.channel_num > 0:
            index = list(range(C))
            random.shuffle(index)
            index = index[:self.channel_num]
            feature = feature.index_select(index=torch.tensor(index).to(feature.device), dim=-1)
            C = self.channel_num

        input_x = feature

        # B T N C -> B C N T
        # feature = feature.transpose(1, 3).contiguous().view(B*C, N, -1)
        feature = feature.transpose(1, 3)  # B, C, N, T

        masks = self.tcn_tac(feature)  # B*C, num_spk, N, T
        masks = masks.transpose(2, 3)  # B*C, num_spk, T, N
        masks = masks.contiguous().view(B, C, self.num_spk, T, N).permute(0, 2, 3, 4, 1) # B num_spk, T, N, C
        masks = masks.unbind(dim=1)  # List[B, T, N, C]

        masked = [torch.sum(input_x * m, dim=3) for m in masks] # List[B, T, N]

        others = OrderedDict(
            zip(["mask_spk{}".format(i + 1) for i in range(len(masks))], masked)
        )

        return masked, ilens, others

    @property
    def num_spk(self):
        return self._num_spk
