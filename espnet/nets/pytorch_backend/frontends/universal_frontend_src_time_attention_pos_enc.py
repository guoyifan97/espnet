from distutils.version import LooseVersion
from typing import Type, Any, Callable, Union, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_complex import functional as FC
from torch_complex.tensor import ComplexTensor

import espnet.nets.pytorch_backend.frontends._complexLayers as c_nn
import espnet.nets.pytorch_backend.frontends._complexFunctions as c_F


class Universal_Frontend_Src_Time_Attention_Pos_Enc(nn.Module):
    '''
        A univeral frontend,
        can be used for monochannel audio
        and multichannel audio,
        Complex-valued CNN and self-attention 
        used as the beamformer.

        如果设置use_residual=False,
        则这一版的最后一个Attention module 的forward Layer
        将没有residual，且不用relu.

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
        use_sub_sampling: bool = False,
        sub_sampling_type: str = "conv2d",
        downconv_type: str = "conv1d",
        use_time_high_dim_as_v: bool = False,
        use_pos_embed: bool = True,
        pos_embed_type: str = "2D",
        beamformer_type: str = "aoliao",
        ):
        
        super().__init__()

        assert (use_sub_sampling and len(conv_layer_list) == 0) or not use_sub_sampling, "cannot use subsampling with conv_layers"
        assert sub_sampling_type in ["conv2d", "conv2d6", "conv2d8",], ""
        
        self.conv_layer_list = conv_layer_list
        self.inplane = inplane
        self.att_feat = att_feat
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
            if downconv_type == "conv1d":
                self.downconv = c_nn.ComplexConv2d(temp_inplane, 1, kernel_size=1)
            elif downconv_type == "stack_linear":
                self.downconv = c_nn.ComplexLinear(temp_inplane * fft_feat, fft_feat)
            else:
                raise ValueError("The downconv type We now support are 'conv1d' and 'stack_linear' but get {}".format(downconv_type))
            self.downconv_type = downconv_type
            

        if use_sub_sampling:
            print(f"We use {sub_sampling_type} to subsampling")
            if sub_sampling_type=="conv2d":
                self.sub_sample_layer = c_nn.ComplexConv2dSubsampling(fft_feat, fft_feat, dropout_rate)
            elif sub_sampling_type=="conv2d6":
                self.sub_sample_layer = c_nn.ComplexConv2dSubsampling6(fft_feat, fft_feat, dropout_rate)
            elif sub_sampling_type=="conv2d8":
                self.sub_sample_layer = c_nn.ComplexConv2dSubsampling8(fft_feat, fft_feat, dropout_rate)

        self.use_time_high_dim_as_v = use_time_high_dim_as_v
        print(f"We use {reduce_method} as reduce method")
        if use_pos_embed:
            print(f"We use {pos_embed_type} as pos embedding method")

        self.beamforming_module = BeamformingModule(
            n_blocks=n_att_blocks,
            n_head=n_att_head,
            in_channel_feat=fft_feat,
            in_time_feat=fft_feat,
            att_feat=att_feat,
            forward_layer="linear",
            dropout_rate=dropout_rate,
            reduce_method=reduce_method,
            use_residual=use_residual,
            use_time_high_dim_as_v=use_time_high_dim_as_v,
            use_pos_embed = use_pos_embed,
            pos_embed_type = pos_embed_type,
            module_type = beamformer_type,
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
        del data
        B, C, T, F = x_r.shape 
        # if self.training:
        #     channel = int(torch.randint(C,(1,)))
        # else:
        #     channel = 0
        
        # Time Module BCTF -> B1TF
        # x_time_r, x_time_i = x_r[:, channel:channel+1], x_i[:, channel:channel+1]
        x_time_r, x_time_i = x_r.mean(1).unsqueeze(1), x_i.mean(1).unsqueeze(1)

        # Beamformer / ResNet: B 1 T F -> B C' T F
        for conv in self.conv_layers:
            x_time_r, x_time_i = conv(x_time_r, x_time_i)

        # B C' T F -> B 1 T F 
        if hasattr(self, "downconv"):
            if getattr(self, "downconv_type", False) == "stack_linear":
               # B C' T F -> B 1 T C'*F
               x_time_r, x_time_i = x_time_r.transpose(1, 2).contiguous().view(B, 1, T, -1), x_time_i.transpose(1,2).contiguous().view(B, 1, T, -1)
            
            x_time_r, x_time_i = self.downconv(x_time_r, x_time_i)
        
        # B 1 T F -> B T F
        if hasattr(self, "sub_sample_layer"):
            x_time_r, x_time_i, ilens = self.sub_sample_layer(x_time_r, x_time_i, ilens)
        
        # B 1 T F or B T F -> BCTF
        x_time_r, x_time_i = x_time_r.contiguous().view(B, 1, T, F).expand(B, C, T, F), x_time_i.contiguous().view(B, 1, T, F).expand(B, C, T, F)
        
        # mask B C C and B T T
        x_channel_mask = None
        if self.use_time_high_dim_as_v:
            x_time_mask = make_non_pad_mask(ilens, torch.empty(B, T, ilens.max()).to(x_time_r.device)) * make_non_pad_mask(ilens, torch.empty(B, T, ilens.max()).to(x_time_r.device), -2)
        else:
            x_time_mask = make_non_pad_mask(ilens, torch.empty(B, T, T).to(x_time_r.device)) * make_non_pad_mask(ilens, torch.empty(B, T, T).to(x_time_r.device), -2)
            
        # BCTF -> BTF
        x_r, x_i = self.beamforming_module(x_r, x_i, x_channel_mask, x_time_r, x_time_i, x_time_mask)
        del x_time_r, x_time_i, x_channel_mask, x_time_mask
        out = ComplexTensor(x_r, x_i)
        del x_r,x_i

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
        self.relu = c_nn.ComplexLeakyReLU(inplace=True)
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
        del x_r, x_i
        out = self.conv1(*out)
        out = self.bn1(*out) # actually layernorm
        out = self.relu(*out)
        
        out = c_F.complex_zero_pad(out[0], out[1], (self.kernel_size-1, 0, self.dilation[1]*(self.kernel_size-1), 0))
        out = self.conv2(*out)
        x_r, x_i = self.bn2(*out)
        
        del out

        if self.downsample != None:
            identity_r, identity_i = self.downsample(identity_r, identity_i)

        x_r = x_r + identity_r
        x_i = x_i + identity_i
        del identity_i, identity_r
        x_r, x_i = self.relu(x_r, x_i)

        return x_r, x_i


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


class BeamformingModule(nn.Module):
    def __init__(
        self, 
        n_blocks: int = 2, 
        n_head: int = 4,
        in_channel_feat: int = 257,
        in_time_feat: int = 257,
        att_feat: int = 256,
        forward_layer: str = "linear",
        dropout_rate: float = 0,
        reduce_method: str = "mask",
        use_residual: bool = True,
        use_time_high_dim_as_v: bool = False,
        use_pos_embed: bool = True,
        pos_embed_type: str = "2D",
        module_type: str = "aoliao",
        ):
        """ Channel Wise Attention Block 
        
        :param int n_blocks: number of attention blocks
        :param int n_head: number of attention's head
        :param int in_channel_feat: dimension of the input channel feature
        :param int in_time_feat: dimensiong of the input time feature
        :param int att_feat: dimension of the hidden attention feat
        :param str forward_layer: the type of forward layer
        :param float dropout_rate: the dropout rate
        :param str reduce_method: the method to generate mono channel feat
        """
        super().__init__()

        if in_channel_feat != att_feat:
            self.channel_before_att_ln = c_nn.ComplexLinear(in_channel_feat, att_feat)
            self.after_att_ln = c_nn.ComplexLinear(att_feat, in_channel_feat)
        
        if in_time_feat != att_feat:
            self.time_before_att_ln = c_nn.ComplexLinear(in_time_feat, att_feat)

        if use_pos_embed:
            if pos_embed_type == "2D":
                self.channel_pos_embed = c_nn.ComplexPositionalEncoding2D(att_feat, dropout_rate)
                self.time_pos_embed = c_nn.ComplexPositionalEncoding2D(att_feat, dropout_rate)
            elif pos_embed_type == "omit_channel":
                self.channel_pos_embed = c_nn.ComplexPositionalEncoding2D(att_feat, dropout_rate, omit_channel=True)
                self.time_pos_embed = c_nn.ComplexPositionalEncoding2D(att_feat, dropout_rate, omit_channel=True)
        else:
            self.channel_pos_embed = FakeModule()
            self.time_pos_embed = FakeModule()
        
        self.time_before_norm = c_nn.ComplexLayerNorm(1, affine=False)
        
        self.att_module = nn.ModuleList()
        for num in range(n_blocks):
            self.att_module.append(
                BeamformingLayer(
                    n_head=n_head,
                    att_in_feat=att_feat,
                    att_out_feat=att_feat,
                    forward_layer=forward_layer,
                    dropout_rate=dropout_rate,
                    use_residual=True,
                    use_time_high_dim_as_v=use_time_high_dim_as_v,
                    )
            )
        
        assert reduce_method in ["mask", "mean"], f"reduce_method should be 'mask' or 'mean', but get {reduce_method}"
        self.reduce_method = reduce_method
        self.norm = c_nn.ComplexLayerNorm(1, affine=False)


    def forward(self, x_channel_r, x_channel_i, x_channel_mask, x_time_r, x_time_i, x_time_mask):
        """

        Args:
            x_channel_r, x_channel_i: (batch, channel, time, att_feat)
            x_channel_mask: None or (batch, 1, channel) or (batch, channel, channel)
            x_time_r, x_time_i: (batch, C, time, att_feat)
            x_time_mask: (batch, 1, time) or (batch, time, time)
        return 
            out_r, out_i: (batch, time, att_out_feat)

        """
    
        if hasattr(self,"channel_before_att_ln"):
            out_channel_r, out_channel_i = self.channel_before_att_ln(x_channel_r, x_channel_i)
        else:
            out_channel_r, out_channel_i = x_channel_r, x_channel_i
        
        if hasattr(self,"time_before_att_ln"):
            x_time_r, x_time_i = self.time_before_att_ln(x_time_r, x_time_i)
        
        # position embedding
        out_channel_r, out_channel_i = self.channel_pos_embed(out_channel_r, out_channel_i)

        x_time_r, x_time_i = self.time_before_norm(*self.time_pos_embed(x_time_r, x_time_i))

        x_time_r, x_time_i = x_time_r.transpose(1,2), x_time_i.transpose(1,2)

        for layer in self.att_module:
            out_channel_r, out_channel_i = layer(out_channel_r, out_channel_i, x_channel_mask, x_time_r, x_time_i, x_time_mask)
        
        del x_time_r, x_time_i, x_channel_mask, x_time_mask

        out_channel_r, out_channel_i = self.norm(out_channel_r, out_channel_i)
        
        if hasattr(self,"after_att_ln"):
            out_channel_r, out_channel_i = self.after_att_ln(out_channel_r, out_channel_i)

        if self.reduce_method == "mask":
            out_channel_r, out_channel_i = out_channel_r*x_channel_r + out_channel_i*x_channel_i, out_channel_r*x_channel_i - out_channel_i*x_channel_r # conj of mask multiply input
        
        del x_channel_r, x_channel_i

        out_channel_r, out_channel_i = out_channel_r.mean(dim=1), out_channel_i.mean(dim=1)
            
        return out_channel_r, out_channel_i


class BeamformingLayer(nn.Module):
    def __init__(
        self,
        n_head: int = 4,
        att_in_feat: int = 256,
        att_out_feat: int = 256,
        forward_layer: str = "linear",
        dropout_rate: float = 0,
        use_residual: bool = True,
        use_relu: bool = True,
        use_time_high_dim_as_v: bool = False,
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
        self.dropout = c_nn.ComplexDropout(dropout_rate)

        self.channel_wise_att_module = c_nn.ComplexMultiHeadedAttention(n_head, att_in_feat, att_in_feat, dropout_rate)
        self.norm1 = c_nn.ComplexLayerNorm(1, affine=False) # we should fix this bug that we have to know the channel even when we don't use "affine"

        # if att_in_feat != att_out_feat:
        #     self.att_residual_ln = c_nn.ComplexLinear(att_in_feat, att_out_feat)
        self.use_time_high_dim_as_v = use_time_high_dim_as_v
        self.time_wise_att_module = c_nn.ComplexMultiHeadedAttention(n_head, att_in_feat, att_out_feat, dropout_rate)
        self.norm2 = c_nn.ComplexLayerNorm(1, affine=False)


        self.forward_layer = c_nn.ComplexSequential(
            c_nn.ComplexLinear(att_out_feat, 2048),
            c_nn.ComplexLeakyReLU(inplace=True),
            c_nn.ComplexDropout(dropout_rate),
            c_nn.ComplexLinear(2048, att_out_feat),
            )
        
        self.norm3 = c_nn.ComplexLayerNorm(1, affine=False)
        
        
        

    def forward(self, x_channel_r, x_channel_i, x_channel_mask, x_time_r, x_time_i, x_time_mask):
        """
        :param torch.Tensor x_channel_r, x_channel_i: (batch, channel, time, att_in_feat)
        :param torch.Tensor x_time_r, x_time_i: (batch, time, channel, att_in_feat)
        x_channel_mask: (batch, max_channel, max_channel) or None
        x_time_mask: (batch, max_time_in, max_time_in)
        :return torch.Tensor x_r, x_i: (batch, channel, time, att_out_feat)

        """
        
        # Channel Wise Attention
        residual_r, residual_i = x_channel_r, x_channel_i

        x_channel_r, x_channel_i = self.norm1(x_channel_r, x_channel_i)

        x_channel_r, x_channel_i = self.dropout(*self.channel_wise_att_module([x_channel_r, x_channel_i], [x_channel_r, x_channel_i], [x_channel_r, x_channel_i], x_channel_mask))
        del x_channel_mask
        x_channel_r, x_channel_i = residual_r + x_channel_r, residual_i + x_channel_i

        del residual_r, residual_i
        # Src Attention in temporal dimension 
        # B C T F -> B T C F
        x_channel_r, x_channel_i = x_channel_r.transpose(1,2), x_channel_i.transpose(1,2)
        residual_r, residual_i = x_channel_r, x_channel_i
        
        # if hasattr(self, "att_residual_ln"):
        #     residual_r, residual_i = self.att_residual_ln(residual_r, residual_i)

        x_channel_r, x_channel_i = self.norm2(x_channel_r, x_channel_i)
        if self.use_time_high_dim_as_v:
            x_channel_r, x_channel_i = self.dropout(*self.time_wise_att_module([x_channel_r, x_channel_i], [x_time_r, x_time_i], [x_time_r, x_time_i], x_time_mask))
        else:
            x_channel_r, x_channel_i = self.dropout(*self.time_wise_att_module([x_time_r, x_time_i], [x_channel_r, x_channel_i], [x_channel_r, x_channel_i], x_time_mask))
        
        del x_time_r, x_time_i, x_time_mask

        x_channel_r, x_channel_i = residual_r + x_channel_r, residual_i + x_channel_i


        # Forward Layer
        residual_r, residual_i = x_channel_r, x_channel_i

        x_channel_r, x_channel_i = self.norm3(x_channel_r, x_channel_i)
        x_channel_r, x_channel_i = self.dropout(*self.forward_layer(x_channel_r, x_channel_i))
        
        x_channel_r, x_channel_i = residual_r + x_channel_r, residual_i + x_channel_i

        return x_channel_r.transpose(1,2), x_channel_i.transpose(1,2)

          
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

def make_pad_mask(lengths, xs=None, length_dim=-1):
    
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
    return mask

