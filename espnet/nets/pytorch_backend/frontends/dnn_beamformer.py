from distutils.version import LooseVersion
from typing import Tuple, List

import torch
from torch.nn import functional as F
from torch_complex import functional as FC
from torch_complex.tensor import ComplexTensor

from espnet.nets.pytorch_backend.frontends.beamformer import apply_beamforming_vector
# from espnet.nets.pytorch_backend.frontends.beamformer import get_mvdr_vector
from espnet.nets.pytorch_backend.frontends.beamformer import (
    get_power_spectral_density_matrix,  # noqa: H301
)
from espnet.nets.pytorch_backend.frontends.mask_estimator import MaskEstimator
from espnet.nets.pytorch_backend.frontends._complexLayers import ComplexLayerNorm, ComplexLeakyReLU, ComplexLinear, ComplexSequential
from espnet.nets.pytorch_backend.frontends.universal_frontend import BasicBlock

import espnet.nets.pytorch_backend.frontends._complexLayers as c_nn
import espnet.nets.pytorch_backend.frontends._complexFunctions as c_F

is_torch_1_2_plus = LooseVersion(torch.__version__) >= LooseVersion("1.2.0")
is_torch_1_3_plus = LooseVersion(torch.__version__) >= LooseVersion("1.3.0")


def get_mvdr_vector(
    psd_s: ComplexTensor,
    psd_n: ComplexTensor,
    reference_vector: torch.Tensor,
    eps: float = 1e-15,
) -> ComplexTensor:
    """Return the MVDR(Minimum Variance Distortionless Response) vector:

        h = (Npsd^-1 @ Spsd) / (Tr(Npsd^-1 @ Spsd)) @ u

    Reference:
        On optimal frequency-domain multichannel linear filtering
        for noise reduction; M. Souden et al., 2010;
        https://ieeexplore.ieee.org/document/5089420

    Args:
        psd_s (ComplexTensor): (..., F, C, C)
        psd_n (ComplexTensor): (..., F, C, C)
        reference_vector (torch.Tensor): (..., C)
        eps (float):
    Returns:
        beamform_vector (ComplexTensor)r: (..., F, C)
    """
    # Add eps
    C = psd_n.size(-1)
    eye = torch.eye(C, dtype=psd_n.dtype, device=psd_n.device)
    shape = [1 for _ in range(psd_n.dim() - 2)] + [C, C]
    eye = eye.view(*shape)
    psd_n += eps * eye
    
    try:
        reg_coeff_tensor = ComplexTensor(torch.rand_like(psd_n.real),
        torch.rand_like(psd_n.real))*1e-5
        temp = torch.max(psd_n.real.abs().max(), psd_n.imag.abs().max())
        psd_n = psd_n / temp * 100
        psd_s = psd_s / temp * 100
        psd_n = psd_n + reg_coeff_tensor
        psd_n_i = psd_n.inverse()
    
    except:
        print("THIS M WRONG",psd_n)
        try:
            reg_coeff_tensor = ComplexTensor(torch.rand_like(psd_n.real),
            torch.rand_like(psd_n.real))*1e-2
            psd_n = psd_n/10e+4
            psd_s = psd_s/10e+4
            if psd_n.real.device.index==0:
                print("beamformer.py nan_test psd_n", torch.tensor([torch.isnan(psd_n.real).any(), torch.isnan(psd_n.imag).any()]).any())
                if torch.tensor([torch.isnan(psd_n.real).any(), torch.isnan(psd_n.imag).any()]).any():
                    print("beamformer.py nan_test psd_n: WRONG", psd_n)
                print("beamformer.py nan_test reg_coeff", torch.tensor([torch.isnan(reg_coeff_tensor.real).any(), torch.isnan(reg_coeff_tensor.imag).any()]).any())
                if torch.tensor([torch.isnan(reg_coeff_tensor.real).any(), torch.isnan(reg_coeff_tensor.imag).any()]).any():
                    print("beamformer.py nan_test reg_coeff: WRONG", reg_coeff)
            psd_n += reg_coeff_tensor
            psd_n_i = psd_n.inverse()
            if psd_n.real.device.index==0:
                print("beamformer.py nan_test psd_n + reg", torch.tensor([torch.isnan(psd_n.real).any(), torch.isnan(psd_n.imag).any()]).any())
                if torch.tensor([torch.isnan(psd_n.real).any(), torch.isnan(psd_n.imag).any()]).any():
                    print("beamformer.py nan_test psd_n + reg: WRONG", psd_n)
                print("beamformer.py nan_test psd_n_i", torch.tensor([torch.isnan(psd_n_i.real).any(), torch.isnan(psd_n_i.imag).any()]).any())
                if torch.tensor([torch.isnan(psd_n_i.real).any(), torch.isnan(psd_n_i.imag).any()]).any():
                    print("beamformer.py nan_test psd_n_i: WRONG", psd_n_i)
        except:
            # try:
            #     reg_coeff_tensor = ComplexTensor(torch.rand_like(psd_n.real),
            #     torch.rand_like(psd_n.real))*1e-1
            #     psd_n = psd_n/10e+10
            #     psd_s = psd_s/10e+10
            #     psd_n += reg_coeff_tensor
            #     psd_n_i = psd_n.inverse()
            # except:
            
            reg_coeff_tensor = ComplexTensor(torch.rand_like(psd_n.real),
            torch.rand_like(psd_n.real))*1e-6
            temp = torch.max(psd_n.real.abs().max(), psd_n.imag.abs().max())
            psd_n = psd_n / temp
            psd_s = psd_s / temp
            if psd_n.real.device.index==0:
                print("beamformer.py nan_test psd_n", torch.tensor([torch.isnan(psd_n.real).any(), torch.isnan(psd_n.imag).any()]).any())
                if torch.tensor([torch.isnan(psd_n.real).any(), torch.isnan(psd_n.imag).any()]).any():
                    print("beamformer.py nan_test psd_n: WRONG", psd_n)
                print("beamformer.py nan_test reg_coeff", torch.tensor([torch.isnan(reg_coeff_tensor.real).any(), torch.isnan(reg_coeff_tensor.imag).any()]).any())
                if torch.tensor([torch.isnan(reg_coeff_tensor.real).any(), torch.isnan(reg_coeff_tensor.imag).any()]).any():
                    print("beamformer.py nan_test reg_coeff: WRONG", reg_coeff)
            psd_n += reg_coeff_tensor
            psd_n_i = psd_n.inverse()
            if psd_n.real.device.index==0:
                print("beamformer.py nan_test psd_n + reg", torch.tensor([torch.isnan(psd_n.real).any(), torch.isnan(psd_n.imag).any()]).any())
                if torch.tensor([torch.isnan(psd_n.real).any(), torch.isnan(psd_n.imag).any()]).any():
                    print("beamformer.py nan_test psd_n + reg: WRONG", psd_n)
                print("beamformer.py nan_test psd_n_i", torch.tensor([torch.isnan(psd_n_i.real).any(), torch.isnan(psd_n_i.imag).any()]).any())
                if torch.tensor([torch.isnan(psd_n_i.real).any(), torch.isnan(psd_n_i.imag).any()]).any():
                    print("beamformer.py nan_test psd_n_i: WRONG", psd_n_i)

    numerator = FC.einsum('...ec,...cd->...ed', [psd_n_i, psd_s])

    ws = numerator / (FC.trace(numerator)[..., None, None] + eps)
    # h: (..., F, C_1, C_2) x (..., C_2) -> (..., F, C_1)
    beamform_vector = FC.einsum("...fec,...c->...fe", [ws, reference_vector])
    return beamform_vector

class Mono_Process(torch.nn.Module):
    def __init__(
        self,
        n_block: int = 4,
        idim: int = 257,
        hdim: int = 257,
    ):
        super().__init__()

        self.bn = ComplexLayerNorm(1, affine=False)
        self.relu = ComplexLeakyReLU()
        assert n_block > 1, f"{n_block}"
        self.ln_block = torch.nn.ModuleList()
        self.ln_block.append(
                ComplexSequential(
                    ComplexLinear(idim, hdim),
                    ComplexLeakyReLU(),
                    ComplexLinear(hdim, hdim),
                )
            )
        for i in range(n_block-2):
            self.ln_block.append(
                ComplexSequential(
                    ComplexLinear(hdim, hdim),
                    ComplexLeakyReLU(),
                    ComplexLinear(hdim, hdim),
                )
            )
        self.ln_block.append(
            ComplexSequential(
                    ComplexLinear(hdim, hdim),
                    ComplexLeakyReLU(),
                    ComplexLinear(hdim, idim),
                )
        )
    
    def forward(self, data):
        x_r, x_i = data.real, data.imag

        for layer in self.ln_block:
            residual_r, residual_i = x_r, x_i
            x_r, x_i = layer(x_r, x_i)
            x_r, x_i = self.bn(x_r, x_i)
            x_r, x_i = self.relu(residual_r + x_r, residual_i + x_i)
        
        return ComplexTensor(x_r, x_i)

class Mono_Process_V2(torch.nn.Module):
    def __init__(
        self,
        n_block: int = 4,
        idim: int = 257,
        hdim: int = 257,
        conv_layer_dilation: List[int] = 8*[1],
    ):
        super().__init__()
        
        # ResNet Beamformer related
        conv_layer_list = [2,4,8,16]
        self.conv_layers = torch.nn.ModuleList()
        temp_inplane = 1
        for index, i in enumerate(conv_layer_list):
            if temp_inplane!=i:
                self.conv_layers.append(BasicBlock(temp_inplane, i, kernel_size=3, downsample=c_nn.ComplexSequential(
                    c_nn.ComplexConv2d(temp_inplane, i, kernel_size=1),
                    c_nn.ComplexLayerNorm(i),
                    ),
                    dilation=conv_layer_dilation[index*2: index*2+2],
                    ))
            else:
                self.conv_layers.append(BasicBlock(temp_inplane, i, kernel_size=3, dilation=conv_layer_dilation[index*2: index*2+2],))
            temp_inplane = i


        if temp_inplane != 1:
            self.downconv = c_nn.ComplexConv2d(temp_inplane, 1, kernel_size=1)

        self.bn = ComplexLayerNorm(1, affine=False)
        self.relu = ComplexLeakyReLU()
        assert n_block > 1, f"{n_block}"
        self.ln_block = torch.nn.ModuleList()
        self.ln_block.append(
                ComplexSequential(
                    ComplexLinear(idim, hdim),
                    ComplexLeakyReLU(),
                    ComplexLinear(hdim, hdim),
                )
            )
        for i in range(n_block-2):
            self.ln_block.append(
                ComplexSequential(
                    ComplexLinear(hdim, hdim),
                    ComplexLeakyReLU(),
                    ComplexLinear(hdim, hdim),
                )
            )
        self.ln_block.append(
            ComplexSequential(
                    ComplexLinear(hdim, hdim),
                    ComplexLeakyReLU(),
                    ComplexLinear(hdim, idim),
                )
        )
    
    def forward(self, data):
        # BT1F -> BTF -> B1TF
        data = data.squeeze(2).unsqueeze(1)
        x_r, x_i = data.real, data.imag

        # Beamformer / ResNet: B 1 T F -> B C' T F
        for conv in self.conv_layers:
            x_r, x_i = conv(x_r, x_i)

        # B C' T F -> B 1 T F 
        if hasattr(self, "downconv"):
            x_r, x_i = self.downconv(x_r, x_i)


        for layer in self.ln_block:
            residual_r, residual_i = x_r, x_i
            x_r, x_i = layer(x_r, x_i)
            x_r, x_i = self.bn(x_r, x_i)
            x_r, x_i = self.relu(residual_r + x_r, residual_i + x_i)
        
        mask = ComplexTensor(x_r, x_i)
        
        return (mask * data).squeeze(1)



class DNN_Beamformer(torch.nn.Module):
    """DNN mask based Beamformer

    Citation:
        Multichannel End-to-end Speech Recognition; T. Ochiai et al., 2017;
        https://arxiv.org/abs/1703.04783

    """

    def __init__(
        self,
        bidim,
        btype="blstmp",
        blayers=3,
        bunits=300,
        bprojs=320,
        bnmask=2,
        dropout_rate=0.0,
        badim=320,
        ref_channel: int = -1,
        beamformer_type="mvdr",
        mono_process: bool = False,
        mono_process_type: int = 1,
    ):
        super().__init__()
        self.mask = MaskEstimator(
            btype, bidim, blayers, bunits, bprojs, dropout_rate, nmask=bnmask
        )
        self.ref = AttentionReference(bidim, badim)
        self.ref_channel = ref_channel

        self.nmask = bnmask
        if mono_process:
            print(f"We shall have mono_process module: V{mono_process_type}")
            if mono_process_type==1:
                self.mono_mask = Mono_Process()
            elif mono_process_type==2:
                self.mono_process = Mono_Process_V2()
            else:
                raise ValueError(
                    f"Only support two kinds of Beamformer 1 and two, get {mono_process_type}"
                )
        if beamformer_type != "mvdr":
            raise ValueError(
                "Not supporting beamformer_type={}".format(beamformer_type)
            )
        self.beamformer_type = beamformer_type

    def forward(
        self, data: ComplexTensor, ilens: torch.LongTensor
    ) -> Tuple[ComplexTensor, torch.LongTensor, ComplexTensor]:
        """The forward function

        Notation:
            B: Batch
            C: Channel
            T: Time or Sequence length
            F: Freq

        Args:
            data (ComplexTensor): (B, T, C, F)
            ilens (torch.Tensor): (B,)
        Returns:
            enhanced (ComplexTensor): (B, T, F)
            ilens (torch.Tensor): (B,)

        """

        def apply_beamforming(data, ilens, psd_speech, psd_noise):
            # u: (B, C)
            if self.ref_channel < 0:
                u, _ = self.ref(psd_speech, ilens)
                
            else:
                # (optional) Create onehot vector for fixed reference microphone
                u = torch.zeros(
                    *(data.size()[:-3] + (data.size(-2),)), device=data.device
                )
                u[..., self.ref_channel].fill_(1)

            ws = get_mvdr_vector(psd_speech, psd_noise, u)
           
            enhanced = apply_beamforming_vector(ws, data)

            return enhanced, ws
        
        # data (B, T, C, F) -> (B, 1, T, F) -> (B, T, F)
        if data.shape[2] == 1 and (hasattr(self, "mono_mask") or hasattr(self, "mono_process")):
            if hasattr(self, "mono_mask"):
                enhanced = self.mono_mask(data) * data
                return enhanced.squeeze(2), ilens, None
            else:
                return self.mono_process(data), ilens, None

        # data (B, T, C, F) -> (B, F, C, T)        
        data = data.permute(0, 3, 2, 1)

        # if data.device.index==0:
        #     print("\n\n\ndata shape:(BFCT)", data.shape)


        # mask: (B, F, C, T)
        masks, _ = self.mask(data, ilens)
        
        assert self.nmask == len(masks)

        if self.nmask == 2:  # (mask_speech, mask_noise)
            mask_speech, mask_noise = masks

            psd_speech = get_power_spectral_density_matrix(data, mask_speech)
            psd_noise = get_power_spectral_density_matrix(data, mask_noise)

            
            enhanced, ws = apply_beamforming(data, ilens, psd_speech, psd_noise)

            # (..., F, T) -> (..., T, F)
            enhanced = enhanced.transpose(-1, -2)
            mask_speech = mask_speech.transpose(-1, -3)
        else:  # multi-speaker case: (mask_speech1, ..., mask_noise)
            mask_speech = list(masks[:-1])
            mask_noise = masks[-1]

            psd_speeches = [
                get_power_spectral_density_matrix(data, mask) for mask in mask_speech
            ]
            psd_noise = get_power_spectral_density_matrix(data, mask_noise)

            enhanced = []
            ws = []
            for i in range(self.nmask - 1):
                psd_speech = psd_speeches.pop(i)
                # treat all other speakers' psd_speech as noises
                enh, w = apply_beamforming(
                    data, ilens, psd_speech, sum(psd_speeches) + psd_noise
                )
                psd_speeches.insert(i, psd_speech)

                # (..., F, T) -> (..., T, F)
                enh = enh.transpose(-1, -2)
                mask_speech[i] = mask_speech[i].transpose(-1, -3)

                enhanced.append(enh)
                ws.append(w)

        return enhanced, ilens, mask_speech


class AttentionReference(torch.nn.Module):
    def __init__(self, bidim, att_dim):
        super().__init__()
        self.mlp_psd = torch.nn.Linear(bidim, att_dim)
        self.gvec = torch.nn.Linear(att_dim, 1)

    def forward(
        self, psd_in: ComplexTensor, ilens: torch.LongTensor, scaling: float = 2.0
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        """The forward function

        Args:
            psd_in (ComplexTensor): (B, F, C, C)
            ilens (torch.Tensor): (B,)
            scaling (float):
        Returns:
            u (torch.Tensor): (B, C)
            ilens (torch.Tensor): (B,)
        """
        B, _, C = psd_in.size()[:3]
        assert psd_in.size(2) == psd_in.size(3), psd_in.size()
        # psd_in: (B, F, C, C)
        datatype = torch.bool if is_torch_1_3_plus else torch.uint8
        datatype2 = torch.bool if is_torch_1_2_plus else torch.uint8
        psd = psd_in.masked_fill(
            torch.eye(C, dtype=datatype, device=psd_in.device).type(datatype2), 0
        )
        # psd: (B, F, C, C) -> (B, C, F)
        psd = (psd.sum(dim=-1) / (C - 1)).transpose(-1, -2)

        # Calculate amplitude
        psd_feat = (psd.real ** 2 + psd.imag ** 2) ** 0.5

        # (B, C, F) -> (B, C, F2)
        mlp_psd = self.mlp_psd(psd_feat)
        # (B, C, F2) -> (B, C, 1) -> (B, C)
        e = self.gvec(torch.tanh(mlp_psd)).squeeze(-1)
        u = F.softmax(scaling * e, dim=-1)
        return u, ilens
