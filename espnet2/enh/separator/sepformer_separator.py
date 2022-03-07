from collections import OrderedDict
from distutils.version import LooseVersion
from typing import List
from typing import Tuple
from typing import Union

import torch
from torch_complex.tensor import ComplexTensor

from espnet2.enh.layers.complex_utils import is_complex
from espnet2.enh.layers.sepformer import SepFormer
from espnet2.enh.layers.dprnn import merge_feature
from espnet2.enh.layers.dprnn import split_feature
from espnet2.enh.separator.abs_separator import AbsSeparator
from espnet2.enh.layers.tcn import ChannelwiseLayerNorm

is_torch_1_9_plus = LooseVersion(torch.__version__) >= LooseVersion("1.9.0")


class SepFormerSeparator(AbsSeparator):
    def __init__(
        self,
        input_dim: int,
        num_spk: int = 2,
        encoder_conf: dict = {},
        nonlinear: str = "relu",
        segment_size: int = 250,
    ):
        """Dual-Path RNN (DPRNN) Separator

        Args:
            input_dim: input feature dimension
            num_spk: number of speakers
            nonlinear1: the nonlinear function for mask estimation,
                       select from 'relu', 'tanh', 'sigmoid'
            segment_size: dual-path segment size
        """
        super().__init__()

        self._num_spk = num_spk

        self.segment_size = segment_size

        self.layernorm = ChannelwiseLayerNorm(input_dim)

        self.linear1 = torch.nn.Conv1d(input_dim, input_dim, 1)

        self.sepformer = SepFormer(
            input_size=input_dim,
            num_spk=num_spk,
            **encoder_conf
        )
        
        self.ffw = torch.nn.Sequential(
            torch.nn.Linear(input_dim*num_spk, input_dim*num_spk), 
            {"prelu": torch.nn.PReLU(),
                "sigmoid": torch.nn.Sigmoid(),
                "relu": torch.nn.ReLU(),
                "tanh": torch.nn.Tanh(),}[nonlinear], 
            torch.nn.Linear(input_dim*num_spk, input_dim*num_spk))
        
        if nonlinear not in ("prelu","sigmoid", "relu", "tanh"):
            raise ValueError("Not supporting nonlinear={}".format(nonlinear))

        self.nonlinear = {
            "prelu": torch.nn.PReLU(),
            "sigmoid": torch.nn.Sigmoid(),
            "relu": torch.nn.ReLU(),
            "tanh": torch.nn.Tanh(),
        }[nonlinear]

    def forward(
        self, input: Union[torch.Tensor, ComplexTensor], ilens: torch.Tensor
    ) -> Tuple[List[Union[torch.Tensor, ComplexTensor]], torch.Tensor, OrderedDict]:
        """Forward.

        Args:
            input (torch.Tensor or ComplexTensor): Encoded feature [B, T, N]
            ilens (torch.Tensor): input lengths [Batch]

        Returns:
            masked (List[Union(torch.Tensor, ComplexTensor)]): [(B, T, N), ...]
            ilens (torch.Tensor): (B,)
            others predicted data, e.g. masks: OrderedDict[
                'mask_spk1': torch.Tensor(Batch, Frames, Freq),
                'mask_spk2': torch.Tensor(Batch, Frames, Freq),
                ...
                'mask_spkn': torch.Tensor(Batch, Frames, Freq),
            ]
        """

        # if complex spectrum,
        if is_complex(input):
            feature = abs(input)
        else:
            feature = input

        B, T, N = feature.shape

        feature = feature.transpose(1, 2)  # B, N, T

        feature = self.layernorm(feature) # B, N, T

        feature = self.linear1(feature) # B, N, T

        # h -> h' chunking
        segmented, rest = split_feature(
            feature, segment_size=self.segment_size
        )  # B, N, L, K    (L = segment_size)

        # h' -> h'''
        processed = self.sepformer(segmented, ilens)  # B, N*num_spk, L, K


        # h''' -> h'''' overlap add
        processed = merge_feature(processed, rest)  # B, N*num_spk, T

        processed = processed.transpose(1, 2)  # B, T, N*num_spk

        processed = self.ffw(processed)

        processed = processed.view(B, T, N, self.num_spk)

        masks = self.nonlinear(processed).unbind(dim=3)

        masked = [input * m for m in masks]

        others = OrderedDict(
            zip(["mask_spk{}".format(i + 1) for i in range(len(masks))], masks)
        )

        return masked, ilens, others

    @property
    def num_spk(self):
        return self._num_spk

if __name__ == "__main__":
    encoder_dict = {}
    m = SepFormerSeparator(128, 2)
    # B T N
    xs = torch.rand(3, 10000, 128)
    # B
    ilens = torch.tensor([8000, 9800, 7600])
    masked, ilens, others = m(xs, ilens)
