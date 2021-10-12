from distutils.version import LooseVersion
from typing import Tuple

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

class DNN_Beamformer_V2(torch.nn.Module):
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
    ):
        super().__init__()
        self.mask = MaskEstimator(
            btype, bidim, blayers, bunits, bprojs, dropout_rate, nmask=bnmask
        )
        self.ref = AttentionReference(bidim, badim)
        self.ref_channel = ref_channel

        self.nmask = bnmask

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
                if u.device.index == 0:
                    print("reference u nan test:", torch.isnan(u).any())
                for i, p in enumerate(self.ref.parameters()):
                    if p.device.index==0:
                        print("ref parameters", i, "nantest", torch.isnan(p).any(),torch.isnan(p).all())
                    if i > 8:
                        break
            else:
                # (optional) Create onehot vector for fixed reference microphone
                u = torch.zeros(
                    *(data.size()[:-3] + (data.size(-2),)), device=data.device
                )
                u[..., self.ref_channel].fill_(1)  # fill 1 to the last pos of u

            ws = get_mvdr_vector(psd_speech, psd_noise, u)
            

            enhanced = apply_beamforming_vector(ws, data)

            return enhanced, ws

        # data (B, T, C, F) -> (B, F, C, T)
        data = data.permute(0, 3, 2, 1)

        # if data.device.index==0:
        #     print("\n\n\ndata shape:(BFCT)", data.shape)


        # mask: (B, F, C, T)
        for i, p in enumerate(self.mask.parameters()):
            if p.device.index==0:
                print("mask parameters", i, "nantest", torch.isnan(p).any(),torch.isnan(p).all())
            if i > 10:
                break
        
        masks, _ = self.mask(data, ilens)
        if masks[0].device.index==0:
            print("masks nan test speech (r any all i any all)", torch.isnan(masks[0]).any(),torch.isnan(masks[0]).all(), torch.isnan(masks[1]).any(), torch.isnan(masks[1]).all())
            for i in range(2):
                if torch.isnan(masks[i]).any():
                    print("masks wrong",i,":", masks[i])
        assert self.nmask == len(masks)

        if self.nmask == 2:  # (mask_speech, mask_noise)
            mask_speech, mask_noise = masks

            psd_speech = get_power_spectral_density_matrix(data, mask_speech)
            psd_noise = get_power_spectral_density_matrix(data, mask_noise)
           
            if psd_speech.real.device.index==0:
                print("psd_speech nan test:(r i)", torch.isnan(psd_speech.real).any(), torch.isnan(psd_speech.imag).any())
                print("psd_noise nan test:(r i)", torch.isnan(psd_noise.real).any(), torch.isnan(psd_noise.imag).any())

            enhanced, ws = apply_beamforming(data, ilens, psd_speech, psd_noise)
            if enhanced.device.index==0:
                print("enhaced nan test(before mono mask):", torch.isnan(enhanced.real).any(), torch.isnan(enhanced.imag).any())
                if torch.tensor([torch.isnan(enhanced.real).any(), torch.isnan(enhanced.imag).any()]).any():
                    print("enhaced nan test(before mono mask) WRONG:", enhanced)
            # GUO
            # (B, F, T) -> (B, F, 1, T)
            mono_mask, _ = self.mask(enhanced.unsqueeze(-2), ilens)



            mono_mask_speech, _ = mono_mask   # (B, F, 1, T)
            
            
            
            if mono_mask[0].device.index==0:
                print("mono mask nan test speech (r any all i any all)", torch.isnan(mono_mask[0]).any(),torch.isnan(mono_mask[0]).all(), torch.isnan(mono_mask[1]).any(), torch.isnan(mono_mask[1]).all())
                for i in range(2):
                    if torch.isnan(mono_mask[i]).any():
                        print("mono_mask(speech, noise)",i,":", mono_mask[i]) 
            enhanced = enhanced * mono_mask_speech.detach().squeeze(-2)
            if enhanced.device.index==0:
                print("enhaced nan test(after mono mask):", torch.isnan(enhanced.real).any(), torch.isnan(enhanced.imag).any())
                if torch.tensor([torch.isnan(enhanced.real).any(), torch.isnan(enhanced.imag).any()]).any():
                    print("enhaced nan test(after mono mask) WRONG:", enhanced)

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
