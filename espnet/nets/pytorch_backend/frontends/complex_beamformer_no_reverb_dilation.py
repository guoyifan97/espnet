from distutils.version import LooseVersion
from typing import Type, Any, Callable, Union, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_complex import functional as FC
from torch_complex.tensor import ComplexTensor

import espnet.nets.pytorch_backend.frontends._complexLayers as c_nn
import espnet.nets.pytorch_backend.frontends._complexFunctions as c_F

# def gelu_accurate(x):
#     if not hasattr(gelu_accurate, "_a"):
#         gelu_accurate._a = math.sqrt(2 / math.pi)
#     return (
#         0.5 * x * (1 + torch.tanh(gelu_accurate._a * (x + 0.044715 * torch.pow(x, 3))))
#     )

class Complex_Beamformer(nn.Module):
    '''
        Complex CNN used as the beamformer.
        ResNet18: 2, 2, 2, 2
        ResNet34: 3, 4, 6, 3
        Output enhanced STFT: B x 
    '''

    def __init__(
        self, 
        conv_layer_list: List[int] = [64, 128, 256, 256], 
        inplane=257, 
        num_dereverb_frames=90, 
        num_dereverb_blocks=3, 
        use_dereverb_loss=False, 
        use_clean_loss=False, 
        ratio_reverb: float = 0.4,
        down_sample: bool = False,
        down_sample_layer: List[Tuple[str]] = [(16, 7, 2), (64, 5, 2)],
        use_dilation: bool = True,
        ):
        
        super().__init__()

        self.conv_layer_list = conv_layer_list
        self.inplane = inplane

        # Dereverb related
        self.data_norm = c_nn.ComplexFp42LayerNorm(inplane, affine=False)
        self.num_dereverb_frames = num_dereverb_frames
        self.num_dereverb_blocks = num_dereverb_blocks
        # self.dereverb_block = DereverbBlock(4, inplane, self.num_dereverb_frames, self.num_dereverb_blocks)
        self.use_dereverb_loss = use_dereverb_loss
        self.ratio_reverb = ratio_reverb
        self.down_sample = down_sample
        self.down_sample_list = down_sample_layer
        if self.use_dereverb_loss:
            self.dereverb_fc = c_nn.ComplexLinear(inplane, inplane)
            self.loss_fn_reverb = MSELoss(norm_channel=inplane, norm_func="fp42")

        
        
        temp_inplane = inplane
        
        if self.down_sample:
            # assert use_clean_loss==False, "cannot use both clean MSE and downsample"
            self.down_sample_layers = nn.ModuleList()
            for i, kernel_size, stride in self.down_sample_list:
                self.down_sample_layers.append(
                    c_nn.ComplexConv2d(temp_inplane, i, kernel_size=kernel_size, stride=stride),
                )
                self.down_sample_layers.append(
                    c_nn.ComplexLayerNorm(i),
                )
                self.down_sample_layers.append(
                    c_nn.ComplexReLU(),
                )
                temp_inplane = i

        # ResNet Beamformer related
        self.conv_layers = nn.ModuleList()

        stride, dilation = 1, 1
        len_conv_list = len(self.conv_layer_list)
        for index, i in enumerate(self.conv_layer_list):
            if temp_inplane!=i:
                self.conv_layers.append(Bottleneck(temp_inplane, i, kernel_size=3, stride=(stride,1), dilation=(dilation,1), downsample=c_nn.ComplexSequential(
                    c_nn.ComplexConv2d(temp_inplane, i, kernel_size=1, stride=(stride,1)),
                    c_nn.ComplexLayerNorm(i),
                    ),norm_affine=(index>0)))
            else:
                self.conv_layers.append(Bottleneck(temp_inplane, i, kernel_size=3, stride=(stride,1), dilation=(dilation,1), norm_affine=(index>0)))
            temp_inplane = i
            if use_dilation:
                dilation *= 2
            else:
                if index > len_conv_list-4:
                    stride = 2
                    dilation = 1
                else:
                    dilation *= 2
        self.use_dilation = use_dilation

        # Pool block
        # self.pool_block = PoolBlock(temp_inplane, inplane, kernel_size=[4, 1])
        
        self.downconv = c_nn.ComplexConv2d(temp_inplane, 1, kernel_size=1)

        self.use_clean_loss = use_clean_loss
        if self.use_clean_loss:
            self.loss_fn_clean = MSELoss(norm_channel=inplane, norm_func="fp32")


    def forward(self, data, ilens:torch.Tensor, dereverb_data:ComplexTensor=None, clean_data:ComplexTensor=None):

        """The forward function

        Notation:
            B: Batch
            C: Channel
            T: Time or Sequence length
            F: Freq

        Args:
            data (ComplexTensor): (B, T, C, F)
            clean_data (ComplexTensor): (B, T, F)
            dereverb_data (ComplexTensor): (B, T, C, F)
            ilens (torch.Tensor): (B,)
        Returns:
            enhanced (ComplexTensor): (B, T, F)
            ilens (torch.Tensor): (B,)
            loss (float)

        """

        # BTCF->BCTF
        
        # x_r, x_i = self.data_norm(dereverb_data.real, dereverb_data.imag)
        x_r, x_i = self.data_norm(data.real, data.imag)

        B, C, T, F = x_r.shape
        if self.down_sample:
            for layer in self.down_sample_layers:
                x_r, x_i = layer(x_r, x_i)
            
            for _, kernel_size, stride in self.down_sample_list:
                max_ilen = torch.tensor(float(max(ilens)))
                t_kernel_size = kernel_size if isinstance(kernel_size,int) else kernel_size[0]
                for i in range(ilens.shape[0]):
                    ilens[i] = torch.ceil(ilens[i].float()/stride) if (max_ilen-ilens[i] > t_kernel_size-1) else torch.ceil((max_ilen.float()-t_kernel_size+1)/stride)

        # Dereverb: BFCT -> BFCT
        # x_r, x_i = self.dereverb_block(x_r, x_i)
    
        loss, loss_reverb, loss_clean = None, None, None
        # if self.training and self.use_dereverb_loss:
        if self.use_dereverb_loss:
            assert isinstance(dereverb_data, ComplexTensor), 'there must be dereverb data while "--use-dereverb-loss=True"'
            dereverbed_r, dereverbed_i = self.dereverb_fc(x_r.transpose(1,3), x_i.transpose(1,3))

            loss_reverb = self.loss_fn_reverb(dereverbed_r, dereverbed_i,  \
                dereverb_data.real, dereverb_data.imag, ilens, 0)
            


        # Beamformer / ResNet: BFCT -> BDCT
        len_conv_list = len(self.conv_layer_list)
        for i, conv in enumerate(self.conv_layers):
            x_r, x_i = conv(x_r, x_i)
            if not self.use_dilation and i > len_conv_list-3:
                max_ilen = torch.tensor(float(max(ilens)))
                t_kernel_size = 3
                for i in range(ilens.shape[0]):
                    ilens[i] = torch.ceil(ilens[i].float()/2)
        

        # Pool: BDCT -> BDT -> BTD -> BTF
        x_r, x_i = self.downconv(x_r, x_i)
        x_r, x_i = x_r.squeeze(1), x_i.squeeze(1)
        

        # if self.training and self.use_clean_loss:
        if self.use_clean_loss:
            # clean_data = clean_data / max(clean_data.real.max(), clean_data.imag.max()) * 100
            loss_clean = self.loss_fn_clean(x_r, x_i, clean_data.real, clean_data.imag, ilens)

        out = ComplexTensor(x_r, x_i)

        # raise
        return out, ilens, loss, loss_reverb, loss_clean



class BasicBlock(nn.Module):
    '''
        using n*n kernel, padding n-1 zeros in T and F
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
        
        # if groups != 1 or base_width != 64:
        #     raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        # if dilation > 1:
        #     raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        # self.conv1 = convnxn(inplanes, planes, kernel_size=1, stride=1)
        # self.bn1 = norm_layer(planes)
        self.relu = c_nn.ComplexLeakyReLU()
        self.conv1 = convnxn(inplanes, planes, kernel_size=kernel_size, stride=stride, dilation=(dilation[0],1))
        self.bn1 = norm_layer(planes)
        self.conv2 = convnxn(planes, planes, kernel_size=kernel_size, stride=stride, dilation=(dilation[1],1))
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.kernel_size = kernel_size
        self.dilation = dilation

    def forward(self, x_r, x_i):
        
        # BCTF -> BCTF

        identity_r, identity_i = x_r, x_i

        out = c_F.complex_zero_pad(x_r, x_i, (self.kernel_size-1, 0, self.dilation[0]*(self.kernel_size-1), 0))
        out = self.conv1(*out)
        out = self.bn1(*out) # actually layernorm
        out = self.relu(*out)
        
        out = c_F.complex_zero_pad(out[0], out[1], (self.kernel_size-1, 0, self.dilation[1]*(self.kernel_size-1), 0))
        out = self.conv2(*out)
        x_r, x_i = self.bn2(*out)

        if self.downsample != None:
            identity_r, identity_i = self.downsample(identity_r, identity_i)

        x_r += identity_r
        x_i += identity_i
        out_r, out_i = self.relu(x_r, x_i)

        return out_r, out_i


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 1
    
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        norm_affine: bool = True,
        kernel_size=3,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = c_nn.ComplexLayerNorm

        def convnxn(in_planes, out_planes, kernel_size=4, stride=1, padding=0, groups=1,bias=False, dilation=1):
            """nxn convolution with padding"""
            return c_nn.ComplexConv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, groups=groups, bias=bias, dilation=dilation)

        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = convnxn(inplanes, width, 1)
        self.bn1 = norm_layer(width, affine=norm_affine)
        self.conv2 = convnxn(width, width, kernel_size=kernel_size, stride=stride, groups=groups, dilation=dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = convnxn(width, planes * self.expansion, 1)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = c_nn.ComplexReLU()
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.kernel_size = kernel_size

    def forward(self, x_r, x_i):
        identity_r, identity_i = x_r, x_i

        out = self.conv1(x_r, x_i)
        out = self.bn1(*out)
        out = self.relu(*out)
        
        out = c_F.complex_zero_pad(out[0], out[1], (self.kernel_size-1, 0, self.dilation[0]*(self.kernel_size-1), 0))
        out = self.conv2(*out)
        out = self.bn2(*out)
        out = self.relu(*out)

        out = self.conv3(*out)
        out_r, out_i = self.bn3(*out)

        if self.downsample is not None:
            identity_r, identity_i = self.downsample(identity_r, identity_i)

        out_r += identity_r
        out_i += identity_i
        out = self.relu(out_r, out_i)

        return out[0], out[1]


class DereverbBlock(nn.Module):
    def __init__(self, channel=4, inplane=257, dereverb_frames=90, num_blocks:int=3, transpose=None):
        super().__init__()
        assert num_blocks > 1, "num_block should be larger than 1, got {}".format(num_blocks)
        self.dereverb_frames = dereverb_frames
        self.num_blocks = num_blocks
        self.module_list = nn.ModuleList()
        for i in range(self.num_blocks):
            if i==0:
                # BFCT -> BCFT -> BFCT
                self.module_list.append(
                    c_nn.ComplexConv2d(
                        channel if isinstance(transpose, list) else inplane, channel if isinstance(transpose, list) else inplane, 
                        [1, self.dereverb_frames],
                        transpose=transpose))
                # self.module_list.append(
                #     c_nn.ComplexConv2d(inplane, inplane, 
                #         [1, self.dereverb_frames]))
            else:
                self.module_list.append(
                    c_nn.ComplexConv2d(inplane, inplane, [
                        1, 1
                        ]))
            self.module_list.append(
                c_nn.ComplexLayerNorm(inplane)
                )
            self.module_list.append(
                c_nn.ComplexLeakyReLU()
                )
        
    def forward(self, x_r, x_i):
        '''
        x_r, x_i: B x F x C x T
        '''
        for i in range(self.num_blocks):
            # to make sure the len of input and output are the same 
            if i==0:
                x_r, x_i = c_F.complex_zero_pad(x_r, x_i, (
                    self.dereverb_frames-1, 0
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

        self.linear1 = c_nn.ComplexLinear(inplane, outplane)
        self.norm = c_nn.ComplexFp32LayerNorm(outplane, affine=False)
        self.gelu = c_nn.ComplexLeakyReLU()
        self.linear2 = c_nn.ComplexLinear(outplane, outplane)
    
    def forward(self, x_r, x_i):

        if isinstance(self.kernel_size, int):
            x_r, x_i = c_F.complex_zero_pad(x_r, x_i, (0,self.kernel_size-1))
        # BDCT -> BD1T
        x_r, x_i = self.pool(x_r, x_i)
        # BD1T -> BDT
        x_r, x_i = x_r.squeeze(-2), x_i.squeeze(-2)
        # BDT -> BTD
        x_r, x_i = x_r.transpose(1,2), x_i.transpose(1,2)
        # BTD -> BTF
        x_r, x_i = self.linear1(x_r, x_i)
        x_r, x_i = self.norm(x_r, x_i)
        x_r, x_i = self.gelu(x_r, x_i)

        x_r, x_i = self.linear2(x_r, x_i)

        return x_r, x_i

            
class MSELoss(nn.Module):
    
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', norm_channel:int = None, norm_func:str = None):
        super().__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.reduction = reduction
        self.norm_channel = norm_channel
        self.norm_func = norm_func
        if isinstance(norm_channel, int):
            assert norm_func in ["fp32","fp42"], f"{norm_func} must be fp32 or fp42"
            if norm_func=="fp32":
                self.norm = c_nn.ComplexFp32LayerNorm(norm_channel, affine=False)
            else:
                self.norm = c_nn.ComplexFp42LayerNorm(norm_channel, affine=False)

    def _complex_pow(c_tensor: ComplexTensor, pow_value: float) -> ComplexTensor:
            return ComplexTensor(c_tensor.real.pow(pow_value), c_tensor.imag.pow(pow_value))
    

    def forward(self, x_r, x_i, target_r, target_i, ilens, num_channel=0):
        '''
        x: B T F or B T C F ComplexTensor
        target: B T F or B T C F ComplexTensor
        ilens : B Tensor
        '''
        if isinstance(self.norm, nn.Module):
            target_r, target_i = self.norm(target_r,target_i)
        
            
        if num_channel > 0 :
            mask = make_non_pad_mask(ilens, torch.empty(x_r.shape[0],x_r.shape[1]//num_channel, x_r.shape[-1]), 1)
            mask = torch.cat(num_channel * [mask], dim=1).to(x_r.device).float()
        else:
            mask = make_non_pad_mask(ilens, x_r, 1).to(x_r.device).float()
        
        return 0.5 * ( 
            F.mse_loss(mask * x_r, target_r, size_average=self.size_average, reduce=self.reduce, reduction=self.reduction)
             + F.mse_loss(mask*x_i, target_i, size_average=self.size_average, reduce=self.reduce, reduction=self.reduction) 
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




