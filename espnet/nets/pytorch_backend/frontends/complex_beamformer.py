from distutils.version import LooseVersion
from typing import Tuple, List

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_complex import functional as FC
from torch_complex.tensor import ComplexTensor

import espnet.nets.pytorch_backend.frontends._complexLayers as c_nn
import espnet.nets.pytorch_backend.frontends._complexFunctions as c_F


class Complex_Beamformer(nn.Module):
    '''
        Complex CNN used as the beamformer.
        ResNet18: 2, 2, 2, 2
        ResNet34: 3, 4, 6, 3
        Output enhanced STFT: B x 
    '''

    def __init__(
        self, 
        conv_layer_list: List[int] = 8*[257], 
        inplane=257, 
        num_dereverb_frames=90, 
        num_dereverb_blocks=3, 
        use_dereverb_loss=False, 
        use_clean_loss=False, 
        ratio_reverb:float=0.4
        ):
        
        super().__init__()

        self.conv_layer_list = conv_layer_list
        self.inplane = inplane

        # Dereverb related
        self.num_dereverb_frames = num_dereverb_frames
        self.num_dereverb_blocks = num_dereverb_blocks
        self.dereverb_block = DereverbBlock(inplane, self.num_dereverb_frames, self.num_dereverb_blocks)
        self.use_dereverb_loss = use_dereverb_loss
        self.ratio_reverb = ratio_reverb
        if self.use_dereverb_loss:
            self.dereverb_fc = c_nn.ComplexLinear(inplane, inplane)
            self.loss_fn_reverb = MSELoss()

        # ResNet Beamformer related
        self.conv_layers = nn.ModuleList()
        
        temp_inplane = inplane
        for i in self.conv_layer_list:
            self.conv_layers.append(BasicBlock(temp_inplane, i))
            temp_inplane = i
        
        # Pool block
        self.pool_block = PoolBlock(temp_inplane, inplane, kernel_size=[4, 1])

        self.use_clean_loss = use_clean_loss
        if self.use_clean_loss:
            self.loss_fn_clean = MSELoss()


    def forward(self, data, ilens:torch.Tensor, dereverb_data:ComplexTensor=None, clean_data:ComplexTensor=None):

        """The forward function

        Notation:
            B: Batch
            C: Channel
            T: Time or Sequence length
            F: Freq

        Args:
            data (ComplexTensor): (B, T * C, F)
            clean_data (ComplexTensor): (B, T, F)
            dereverb_data (ComplexTensor): (B, T*C, F)
            ilens (torch.Tensor): (B,)
        Returns:
            enhanced (ComplexTensor): (B, T, F)
            ilens (torch.Tensor): (B,)
            loss (float)

        """

        # BT*CF->BFCT
        
        data = data.contiguous().view(data.shape[0],-1,4,data.shape[-1])
        data = data.permute(0, 3, 2 , 1)
        data = data / max(data.real.max(), data.imag.max()) * 100
        B, F, C, T = data.shape
        ilens = ilens//4

        x_r, x_i = data.real, data.imag
        if x_r.device.index==0:
            print(f"x_r,x_i value mean before dereverb:{x_r.abs().mean()},{x_i.abs().mean()}")

        # Dereverb: BFCT -> BFCT
        x_r, x_i = self.dereverb_block(x_r, x_i)

        if x_r.device.index==0:
            print(f"x_r,x_i value mean after dereverb:{x_r.abs().mean()},{x_i.abs().mean()}")

        loss = None
        if self.training and self.use_dereverb_loss:
            assert isinstance(dereverb_data, ComplexTensor), 'there must be dereverb data while "--use-dereverb-loss=True"'
            # BFCT -> BTCF -> BT*CF
            dereverbed_r, dereverbed_i = self.dereverb_fc(x_r.permute(0,3,2,1).contiguous().view(B,-1,F), x_i.permute(0,3,2,1).contiguous().view(B,-1,F))
            
            if x_r.device.index==0:
                print(f"dereverbed_r,_i value mean after dereverb:{dereverbed_r.abs().mean()},{dereverbed_i.abs().mean()}")
            dereverb_data = dereverb_data / max(dereverb_data.real.max(), dereverb_data.imag.max()) * 100
            loss_reverb = self.loss_fn_reverb(ComplexTensor(dereverbed_r, dereverbed_i),  \
                dereverb_data, ilens, C)
            # loss_reverb = self.loss_fn_reverb(ComplexTensor(dereverbed_r, dereverbed_i),  \
                # dereverb_data.permute(0,3,2,1), ilens)
            loss = loss_reverb


        # Beamformer / ResNet: BFCT -> BDCT
        for conv in self.conv_layers:
            x_r, x_i = conv(x_r, x_i)
        if x_r.device.index==0:
            print(f"x_r,x_i value mean after beamformer:{x_r.abs().mean()},{x_i.abs().mean()}")

        # Pool: BDCT -> BDT -> BTD -> BTF
        x_r, x_i = self.pool_block(x_r, x_i)

        if self.training and self.use_clean_loss:
            clean_data = clean_data / max(clean_data.real.max(), clean_data.imag.max()) * 100
            loss_clean = self.loss_fn_clean(ComplexTensor(x_r, x_i), clean_data, ilens)
            if isinstance(loss, torch.Tensor):
                loss = self.ratio_reverb * loss * float(loss_clean/loss) + (1 - self.ratio_reverb) * loss_clean
            else:
                loss = loss_clean

        out = ComplexTensor(x_r, x_i)
        

        return out, ilens, loss



class BasicBlock(nn.Module):
    '''
        using 4*4 kernel, padding 3 zeros in C and T
    '''
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, kernel_size=4):
        
        super().__init__()

        def convnxn(in_planes, out_planes, kernel_size=4, stride=1, padding=0, groups=1,bias=False, dilation=1):
            """nxn convolution with padding"""
            return c_nn.ComplexConv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, groups=groups, bias=bias, dilation=dilation)

        if norm_layer is None:
            norm_layer = c_nn.ComplexLayerNorm
        
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = convnxn(inplanes, planes,  kernel_size, stride)
        self.bn1 = norm_layer(planes)
        self.relu = c_nn.ComplexReLU()
        self.conv2 = convnxn(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.kernel_size = kernel_size

    def forward(self, x_r, x_i):

        identity_r, identity_i = x_r, x_i
        
        x_r, x_i =  c_F.complex_zero_pad(x_r, x_i, (0, self.kernel_size-1, self.kernel_size-1, 0))
        out = self.conv1(x_r, x_i)
        out = self.bn1(*out) # actually layernorm
        out = self.relu(*out)
        # Pad technic GUO
        out =  c_F.complex_zero_pad(*out, (0, self.kernel_size-1, 0, self.kernel_size-1))
        out = self.conv2(*out)
        x_r, x_i = self.bn2(*out)


        x_r += identity_r
        x_i += identity_i
        out = self.relu(x_r, x_i)

        return out[0], out[1]


class DereverbBlock(nn.Module):
    def __init__(self, inplane=257, dereverb_frames=90, num_blocks:int=3):
        super().__init__()
        assert num_blocks > 1, "num_block should be larger than 1, got {}".format(num_blocks)
        self.dereverb_frames = dereverb_frames
        self.num_blocks = num_blocks
        self.module_list = nn.ModuleList()
        for i in range(self.num_blocks):
            self.module_list.append(
                c_nn.ComplexConv2d(inplane, inplane, [
                    1, self.dereverb_frames//(i+1) if self.dereverb_frames//(i+1) > 15 else 4
                    ]))
            self.module_list.append(
                c_nn.ComplexLayerNorm(inplane)
                )
            self.module_list.append(
                c_nn.ComplexReLU()
                )
        
    def forward(self, x_r, x_i):
        '''
        x_r, x_i: B x F x C x T
        '''
        for i in range(self.num_blocks):
            # to make sure the len of input and output are the same 
            x_r, x_i = c_F.complex_zero_pad(x_r, x_i, (
                self.dereverb_frames//(i+1)-1 if self.dereverb_frames//(i+1) > 15 else 3, 0
                ))
            x_r, x_i = self.module_list[i*3](x_r, x_i)
            x_r, x_i = self.module_list[i*3+1](x_r, x_i)
            x_r, x_i = self.module_list[i*3+2](x_r, x_i)
        return x_r, x_i


class PoolBlock(nn.Module):
    def __init__(self, inplane, outplane=257, kernel_size=4, pool_mode="avg_pool"):
        assert pool_mode in ["conv", "max_pool", "avg_pool"], "pool_mode %s error" % pool_mode
        super().__init__()
        self.inplane = inplane
        self.outplane = outplane
        self.kernel_size = kernel_size
        self.pool_mode = pool_mode
        if pool_mode == "conv":
            self.pool = c_nn.ComplexConv2d(inplane, inplane, kernel_size)
        elif pool_mode == "max_pool":
            self.pool = c_nn.ComplexMaxPool2d(kernel_size)
        else:
            self.pool = c_nn.ComplexAvgPool2d(kernel_size)

        self.linear = c_nn.ComplexLinear(inplane, outplane)
    
    def forward(self, x_r, x_i):

        if isinstance(self.kernel_size, int):
            x_r, x_i = c_F.complex_zero_pad(x_r, x_i, (0,self.kernel_size-1))
        # BDCT -> BD1T
        x_r, x_i = self.pool(x_r, x_i)
        # BD1T -> BDT
        x_r, x_i = x_r.squeeze(-2), x_i.squeeze(-2)
        # BDT -> BTD
        x_r, x_i = x_r.permute(0, 2, 1), x_i.permute(0, 2, 1)
        # BTD -> BTF
        x_r, x_i = self.linear(x_r, x_i)

        return x_r, x_i

            
class MSELoss(nn.Module):
    
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean'):
        super().__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.reduction = reduction

    def _complex_pow(c_tensor: ComplexTensor, pow_value: float) -> ComplexTensor:
            return ComplexTensor(c_tensor.real.pow(pow_value), c_tensor.imag.pow(pow_value))
    

    def forward(self, x, target, ilens, num_channel=0):
        '''
        x: B T F ComplexTensor
        target: B T F ComplexTensor
        ilens : B Tensor
        '''
        if num_channel > 0 :
            mask = make_non_pad_mask(ilens, torch.empty(x.shape[0],x.shape[1]//num_channel, x.shape[-1]), 1)
            mask = torch.cat(num_channel * [mask], dim=1).to(x.device).float()
        else:
            mask = make_non_pad_mask(ilens, x.real, 1).to(x.device).float()
        
        return 0.5 * ( 
            F.mse_loss(mask * x.real, target.real, size_average=self.size_average, reduce=self.reduce, reduction=self.reduction)
             + F.mse_loss(mask*x.imag, target.imag, size_average=self.size_average, reduce=self.reduce, reduction=self.reduction) 
             )



def make_non_pad_mask(lengths, xs=None, length_dim=-1):
    
    if length_dim == 0:
        raise ValueError("length_dim cannot be 0: {}".format(length_dim))

    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    bs = int(len(lengths))
    if xs is None:
        maxlen = int(max(lengths))
    else:
        maxlen = xs.size(length_dim)

    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    if xs is not None:
        assert xs.size(0) == bs, (xs.size(0), bs)

        if length_dim < 0:
            length_dim = xs.dim() + length_dim
        # ind = (:, None, ..., None, :, , None, ..., None)
        ind = tuple(
            slice(None) if i in (0, length_dim) else None for i in range(xs.dim())
        )
        mask = mask[ind].expand_as(xs).to(xs.device)
    return ~mask




