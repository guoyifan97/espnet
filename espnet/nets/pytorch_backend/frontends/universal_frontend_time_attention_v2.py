from distutils.version import LooseVersion
from typing import Type, Any, Callable, Union, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_complex import functional as FC
from torch_complex.tensor import ComplexTensor

import espnet.nets.pytorch_backend.frontends._complexLayers as c_nn
import espnet.nets.pytorch_backend.frontends._complexFunctions as c_F


class Universal_Frontend_Time_Attention_v2(nn.Module):
    '''
        A univeral frontend,
        can be used for monochannel audio
        and multichannel audio,
        Complex-valued CNN and self-attention 
        used as the beamformer.
        
        这个前端，如果设置不使用residual，则attention一个residual都没有，楼上用的v1.
        
        Output enhanced STFT: B x T x F
    '''

    def __init__(
        self, 
        conv_layer_list: List[int] = [4, 8, 16, 32], 
        inplane=1, 
        use_dilation: bool = True,
        n_att_head: int = 4,
        n_att_blocks: int = 2,
        fft_feat: int = 257,
        att_feat: int = 256,
        dropout_rate: float = 0,
        conv_layer_dilation: List[int] = 8*[1],
        reduce_method: str = "mask",
        use_residual: bool = True,
        ):
        
        super().__init__()

        self.conv_layer_list = conv_layer_list
        self.inplane = inplane

        # Dereverb related
        self.data_norm = c_nn.ComplexFp42LayerNorm(inplane, affine=False)
        

        temp_inplane = inplane
        # ResNet Beamformer related
        self.conv_layers = nn.ModuleList()

        for index, i in enumerate(self.conv_layer_list):
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

        print(f"We use {reduce_method} as reduce method")
        self.channel_wise_att_module = ChannelWiseAttentionModule(
            n_blocks=n_att_blocks,
            n_head=n_att_head,
            in_feat=fft_feat,
            att_feat=att_feat,
            forward_layer="linear",
            dropout_rate=dropout_rate,
            reduce_method=reduce_method,
            use_residual=use_residual,
        )


    def forward(self, data, ilens:torch.Tensor):

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

        # BCTF
        x_r, x_i = self.data_norm(data.real, data.imag)

        B, C, T, F = x_r.shape 

        # BCTF -> B*C 1 T F
        x_r, x_i = x_r.contiguous().view(-1, T, F).unsqueeze(1), x_i.contiguous().view(-1, T, F).unsqueeze(1)

        # Beamformer / ResNet: B*C 1 T F -> B*C C' TF
        for conv in self.conv_layers:
            x_r, x_i = conv(x_r, x_i)

        # B*C C' T F -> B*C 1 T F 
        if hasattr(self, "downconv"):
            x_r, x_i = self.downconv(x_r, x_i)
        
        # B*C 1 T F -> BCTF
        x_r, x_i = x_r.contiguous().view(B, C, -1, F), x_i.contiguous().view(B, C, -1, F)
        
        # BCTF -> BTF
        x_r, x_i = self.channel_wise_att_module(x_r, x_i)
        
        out = ComplexTensor(x_r, x_i)

        return out, ilens, None


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
        if isinstance(stride, list):
            assert len(stride)==2
            stride0, stride1 = stride
        else:
            stride0 = stride
            stride1 = stride
        self.relu = c_nn.ComplexLeakyReLU()
        self.conv1 = convnxn(inplanes, planes, kernel_size=kernel_size, stride=(stride0,1), dilation=(dilation[0],1))
        self.bn1 = norm_layer(planes)
        self.conv2 = convnxn(planes, planes, kernel_size=kernel_size, stride=(stride1,1), dilation=(dilation[1],1))
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


class ChannelWiseAttentionModule(nn.Module):
    def __init__(
        self, 
        n_blocks: int = 2, 
        n_head: int = 4,
        in_feat: int = 257,
        att_feat: int = 256,
        forward_layer: str = "linear",
        dropout_rate: float = 0,
        reduce_method: str = "mask",
        use_residual: bool = True,
        ):
        """ Channel Wise Attention Block 
        
        :param int n_blocks: number of attention blocks
        :param int n_head: number of attention's head
        :param int in_feat: dimension of the input feature
        :param int att_feat: dimension of the hidden attention feat
        :param str forward_layer: the type of forward layer
        :param float dropout_rate: the dropout rate
        :param str reduce_method: the method to generate mono channel feat
        """
        super().__init__()

        if in_feat != att_feat:
            self.before_att_ln = c_nn.ComplexLinear(in_feat, att_feat)

        self.att_module = nn.ModuleList()
        for num in range(n_blocks-1):
            self.att_module.append(
                ChannelWiseAttentionLayer(
                    n_head=n_head,
                    att_in_feat=att_feat,
                    att_out_feat=att_feat,
                    forward_layer=forward_layer,
                    dropout_rate=dropout_rate,
                    use_residual=use_residual,
                    )
            )
        self.att_module.append(ChannelWiseAttentionLayer(n_head=n_head, att_in_feat=att_feat, att_out_feat=in_feat, forward_layer=forward_layer, dropout_rate=dropout_rate, use_residual=use_residual))
        
        assert reduce_method in ["mask", "mean"], f"reduce_method should be 'mask' or 'mean', but get {reduce_method}"
        self.reduce_method = reduce_method
        self.norm = c_nn.ComplexLayerNorm(1, affine=False)

    def forward(self, x_r, x_i):
        """
        :param torch.Tensor x_r, x_i: (batch, channel, time, att_feat)
        :return torch.Tensor x_r, x_i: (batch, time, att_out_feat)

        """
        out_r, out_i = x_r, x_i
        if hasattr(self,"before_att_ln"):
            out_r, out_i = self.before_att_ln(x_r, x_i)
        
        for layer in self.att_module:
            out_r, out_i = layer(out_r, out_i)
        
        if self.reduce_method == "mask":
            out_r, out_i = out_r*x_r - out_i*x_i, out_r*x_i + out_i*x_i
        
        out_r, out_i = self.norm(out_r, out_i)

        out_r, out_i = out_r.mean(dim=1), out_i.mean(dim=1)
            
        return out_r, out_i


class ChannelWiseAttentionLayer(nn.Module):
    def __init__(
        self,
        n_head: int = 4,
        att_in_feat: int = 256,
        att_out_feat: int = 257,
        forward_layer: str = "linear",
        dropout_rate: float = 0,
        use_residual: bool = True,
        use_relu: bool = True,
    ):  
        """Channel Wise Attention Layer with residual.

        :param int n_head: the number of head s
        :param int att_in_feat: the number of in features
        :param int att_out_feat: the number of out features
        :param str forward_layer: the type of forward layer
        :param float dropout_rate: dropout rate
        :return torch.Tensor x_r, x_i: (batch, channel, time, att_out_feat)
        """
        super().__init__()
        assert forward_layer=="linear", f"{self.forward_layer}"
        
        self.use_residual = use_residual


        self.channel_wise_att_module = c_nn.ComplexChannelWiseMultiHeadedAttention(n_head, att_in_feat, att_in_feat, dropout_rate)
        self.bn1 = c_nn.ComplexLayerNorm(1, affine=False) # we should fix this bug that we have to know the channel even when we don't use "affine"
        if use_relu:
            self.relu1 = c_nn.ComplexLeakyReLU()
        else:
            self.relu1 = FakeModule()

        if att_in_feat != att_out_feat:
            self.att_residual_ln = c_nn.ComplexLinear(att_in_feat, att_out_feat)
        self.time_wise_att_module = c_nn.ComplexChannelWiseMultiHeadedAttention(n_head, att_in_feat, att_out_feat, dropout_rate)
        self.bn2 = c_nn.ComplexLayerNorm(1, affine=False)
        if use_relu:
            self.relu2 = c_nn.ComplexLeakyReLU()
        else:
            self.relu2 = FakeModule()

        self.forward_layer = c_nn.ComplexSequential(
            c_nn.ComplexLinear(att_out_feat, att_out_feat),
            c_nn.ComplexLeakyReLU(),
            c_nn.ComplexLinear(att_out_feat, att_out_feat),
            )
        
        self.bn3 = c_nn.ComplexLayerNorm(1, affine=False)
        if use_relu:
            self.relu3 = c_nn.ComplexLeakyReLU()
        else:
            self.relu3 = FakeModule()
        
        

    def forward(self, x_r, x_i):
        """
        :param torch.Tensor x_r, x_i: (batch, channel, time, att_in_feat)
        :return torch.Tensor x_r, x_i: (batch, channel, time, att_out_feat)

        """

        # Channel Wise Attention
        if self.use_residual:
            residual_r, residual_i = x_r, x_i

        x_r, x_i = self.bn1(*self.channel_wise_att_module([x_r, x_i], [x_r, x_i], [x_r, x_i]))
        
        if self.use_residual:
            x_r, x_i = x_r + residual_r, x_i + residual_i

        x_r, x_i = self.relu1(x_r, x_i)

        # Attention in Time Dimension
        x_r, x_i = x_r.transpose(1,2), x_i.transpose(1,2)

        if self.use_residual:
            residual_r, residual_i = x_r, x_i
            if hasattr(self, "att_residual_ln"):
                residual_r, residual_i = self.att_residual_ln(residual_r, residual_i)
        
        x_r, x_i = self.bn2(*self.time_wise_att_module([x_r, x_i], [x_r, x_i], [x_r, x_i]))
        
        if self.use_residual:
            x_r, x_i = x_r + residual_r, x_i + residual_i
        
        x_r, x_i = self.relu2(x_r, x_i)

        # Forward Layer
        if self.use_residual:
            residual_r, residual_i = x_r, x_i
        x_r, x_i= self.bn3(*self.forward_layer(x_r, x_i))
        
        if self.use_residual:
            x_r, x_i = x_r + residual_r, x_i + residual_i
        
        x_r, x_i = self.relu3(x_r, x_i)

        return x_r.transpose(1,2), x_i.transpose(1,2)

          
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


class FakeModule(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x_r, x_i):
        return x_r, x_i


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


