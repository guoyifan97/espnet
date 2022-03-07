from distutils.version import LooseVersion
from typing import Type, Any, Callable, Union, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_complex import functional as FC
from torch_complex.tensor import ComplexTensor

import espnet.nets.pytorch_backend.frontends._complexLayers as c_nn
import espnet.nets.pytorch_backend.frontends._complexFunctions as c_F

class CNN_Front_Attention_Mean(nn.Module):
    '''
        Use CNN as the front and use power
        attention or self-attention to get 
        mono output

        Output enhanced STFT: B x T x F
    '''

    def __init__(
        self, 
        conv_layer_list: List[int] = [4, 8, 16, 32], 
        inplane=1, 
        use_dilation: bool = True,
        n_att_head: int = 4,
        n_att_blocks: int = 2,
        n_channel_att_blocks: int = 2,
        n_time_att_blocks: int = 2,
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
        use_pos_embed_in_beamform_layer: bool = False,
        pos_embed_type: str = "2D",
        beamformer_type: str = "shengdai",
        beamformer_return_mask: bool = False,
        use_abs_pos_pred: bool = True,
        num_abs_pos_extracted: int = 1,
        attention_method: str = "power",
        power_att_window_len: int = 2,
        power_att_affine: bool = False,
        power_att_softmax_factor: float = 1.0,
        power_att_auto_grad_softmax_factor: bool = False,
        power_att_use_delta_power_att: bool = False,
        power_att_auto_grad_delta_factor: bool = True,
        power_att_delta_factor: float = 3.0,
        power_att_delta_pos: int = 4,
        abs_pos_loss_factor: float = 50.,
        ):
        
        super().__init__()

        assert (use_sub_sampling and len(conv_layer_list) == 0) or not use_sub_sampling, "cannot use subsampling with conv_layers"
        assert sub_sampling_type in ["conv2d", "conv2d6", "conv2d8",], ""
        assert attention_method in ["power", "channel_wise_att", "seq", "mean"], f"Incompatible attention method: {attention_method}"


        # ResNet Beamformer related        
        self.conv_layer_list = conv_layer_list
        self.inplane = inplane
        self.att_feat = att_feat

        self.data_norm = c_nn.ComplexFp42LayerNorm(inplane, affine=False)
        
        temp_inplane = inplane
        self.conv_layers = nn.ModuleList()
        
        for index, i in enumerate(self.conv_layer_list):
            if isinstance(i, list):
                channel, kernel_size, stride = i
                if temp_inplane!=channel:
                    self.conv_layers.append(BasicBlock(temp_inplane, channel, kernel_size=kernel_size, stride=stride, downsample=c_nn.ComplexSequential(
                        c_nn.ComplexConv2d(temp_inplane, channel, kernel_size=1),
                        c_nn.ComplexLayerNorm(channel),
                        ),
                        dilation=conv_layer_dilation[index*2: index*2+2],
                        ))
                else:
                    self.conv_layers.append(BasicBlock(temp_inplane, channel, kernel_size=kernel_size, stride=stride, dilation=conv_layer_dilation[index*2: index*2+2],))
                temp_inplane = channel
            else:
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
        
        self.after_cnn_norm = c_nn.ComplexLayerNorm(1, affine=False)

        # abc pos information extracted
        self.use_abs_pos_pred = use_abs_pos_pred
        self.num_abs_pos_extracted = num_abs_pos_extracted
        if self.use_abs_pos_pred:
            # pos_feat: B*T F F, ground_truth: 1 F 
            self.ground_truth_abs_pos = torch.arange(fft_feat)[None, :]
            self.abs_pos_norm = c_nn.ComplexLayerNorm(1, affine=False)
            abs_pos_in_feat=0
            for index in range(1,self.num_abs_pos_extracted+1):
                abs_pos_in_feat+=self.conv_layer_list[-index] if isinstance(self.conv_layer_list[-index], int) else self.conv_layer_list[-index][0]
                print(abs_pos_in_feat)
            self.abs_pos_ln_1 = c_nn.ComplexLinear(abs_pos_in_feat, fft_feat)
            self.abs_pos_relu = c_nn.ComplexLeakyReLU(inplace=True)
            self.abs_pos_ln_2 = c_nn.ComplexLinear(fft_feat, fft_feat)
            self.abs_pos_loss_fn = torch.nn.functional.cross_entropy

        self.abs_pos_loss_factor = abs_pos_loss_factor


        # Attention-based beam-select module related
        self.attention_method = attention_method


        self.use_time_high_dim_as_v = use_time_high_dim_as_v

        if self.attention_method == "channel_wise_att":
            print(f"We use {reduce_method} as reduce method")
            if use_pos_embed:
                print(f"We use {pos_embed_type} as pos embedding method")

            if beamformer_return_mask:
                print("We multiply the mask with the orig input")

            self.return_mask = beamformer_return_mask

            self.beamforming_module = BeamformingModule(
                n_blocks=n_att_blocks,
                n_channel_blocks=n_channel_att_blocks,
                n_time_blocks=n_time_att_blocks,
                n_head=n_att_head,
                in_channel_feat=fft_feat,
                in_time_feat=fft_feat,
                att_feat=att_feat,
                forward_layer="linear",
                dropout_rate=dropout_rate,
                reduce_method=reduce_method,
                use_residual=use_residual,
                use_pos_embed = use_pos_embed,
                pos_embed_type = pos_embed_type,
                module_type = beamformer_type,
                return_mask=self.return_mask,
                use_pos_embed_in_beamform_layer=use_pos_embed_in_beamform_layer,
            )
            if use_pos_embed_in_beamform_layer:
                print("We use pos embed on q and k not v")
        elif self.attention_method == "power":
            self.beamforming_module = PowerAttentionModule(
                window_len=power_att_window_len,
                affine=power_att_affine,
                fft_feat=fft_feat,
                softmax_factor=power_att_softmax_factor,
                auto_grad_softmax_factor=power_att_auto_grad_softmax_factor,
                use_delta_power_att=power_att_use_delta_power_att, 
                auto_grad_delta_factor=power_att_auto_grad_delta_factor, 
                delta_factor=power_att_delta_factor, 
                delta_pos=power_att_delta_pos,
                n_blocks=n_att_blocks,
                n_channel_blocks=n_channel_att_blocks,
                n_time_blocks=n_time_att_blocks,
                n_head=n_att_head,
                in_channel_feat=fft_feat,
                in_time_feat=fft_feat,
                att_feat=att_feat,
                forward_layer="linear",
                dropout_rate=dropout_rate,
                reduce_method=reduce_method,
                use_residual=use_residual,
                use_pos_embed = use_pos_embed,
                pos_embed_type = pos_embed_type,
                module_type = beamformer_type,
                use_pos_embed_in_beamform_layer=use_pos_embed_in_beamform_layer,
            )
        elif self.attention_method == "seq":
            print("Warning: When using 'seq' attention_method, the setting of reduce_method will be invalid")
            self.beamforming_module = c_nn.ComplexSequential(
                    BeamformingModule(
                        n_blocks=n_att_blocks,
                        n_channel_blocks=n_channel_att_blocks,
                        n_time_blocks=n_time_att_blocks,
                        n_head=n_att_head,
                        in_channel_feat=fft_feat,
                        in_time_feat=fft_feat,
                        att_feat=att_feat,
                        forward_layer="linear",
                        dropout_rate=dropout_rate,
                        reduce_method=reduce_method,
                        use_residual=use_residual,
                        use_pos_embed = use_pos_embed,
                        pos_embed_type = pos_embed_type,
                        module_type = beamformer_type,
                        return_mask=True,
                        use_pos_embed_in_beamform_layer=use_pos_embed_in_beamform_layer,
                    ),
                    PowerAttentionModule(
                        window_len=power_att_window_len,
                        affine=power_att_affine,
                        fft_feat=fft_feat,
                        softmax_factor=power_att_softmax_factor,
                        auto_grad_softmax_factor=power_att_auto_grad_softmax_factor,
                        use_delta_power_att=power_att_use_delta_power_att, 
                        auto_grad_delta_factor=power_att_auto_grad_delta_factor, 
                        delta_factor=power_att_delta_factor, 
                        delta_pos=power_att_delta_pos,
                        reduce_method=reduce_method,
                    )
            )
        elif self.attention_method == "mean":
            self.beamforming_module = MeanModule(dim=1)

        self.___step = 1


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
        x_time_r, x_time_i = self.data_norm(data.real, data.imag)
        # print(x_time_r.shape)
        if not getattr(self, "return_mask", False):
            del data

        B, C, T, F = x_time_r.shape 

        pos_feat_collection_r, pos_feat_collection_i = [], []
        # Beamformer / ResNet: B C T F -> B C' T F
        for index, conv in enumerate(self.conv_layers):
            x_time_r, x_time_i = conv(x_time_r, x_time_i)
            if self.use_abs_pos_pred and (index > (len(self.conv_layers)-1-self.num_abs_pos_extracted)):
                pos_feat_collection_r.append(x_time_r)
                pos_feat_collection_i.append(x_time_i)
        
        x_time_r, x_time_i = self.after_cnn_norm(x_time_r, x_time_i)

        # cat pos feat to pred the abs position
        # N* BC'TF ->(cat) B N*C' TF -> B*T 1 F N*C'  -> B*T F F
        if self.use_abs_pos_pred:
            # N* BC'TF ->(cat) B N*C' TF
            pos_feat_collection_r, pos_feat_collection_i = torch.cat(pos_feat_collection_r, dim=1), torch.cat(pos_feat_collection_i, dim=1)
            
            #  B N*C' TF -> B T F N*C -> B*T 1 F N*C'
            pos_feat_collection_r, pos_feat_collection_i = \
                pos_feat_collection_r.permute(0,2,3,1).contiguous().view(B*T, 1, F, -1), \
                pos_feat_collection_i.permute(0,2,3,1).contiguous().view(B*T, 1, F, -1)
            
            # B*T 1 F N*C' -> B*T 1 F N*C'
            pos_feat_collection_r, pos_feat_collection_i = self.abs_pos_norm(pos_feat_collection_r, pos_feat_collection_i)
            
            # B*T 1 F N*C' -> B*T F N*C' -> B*T F F
            pos_feat_collection_r, pos_feat_collection_i = self.abs_pos_ln_1(pos_feat_collection_r.squeeze(1), pos_feat_collection_i.squeeze(1))
            
            # B*T F F -> B*T F F
            pos_feat_collection_r, pos_feat_collection_i = self.abs_pos_relu(pos_feat_collection_r, pos_feat_collection_i)
            
            # B*T F F -> B*T F F
            pos_feat_collection_r, pos_feat_collection_i = self.abs_pos_ln_2(pos_feat_collection_r, pos_feat_collection_i)

            pos_feat = (pos_feat_collection_r.pow(2)+pos_feat_collection_i.pow(2)).sqrt()
            del pos_feat_collection_r, pos_feat_collection_i
            # pos_feat: B*T F F -> B*T*F F   ground_truth: 1 F -> B*T F -> B*T*F
            frontend_loss = \
                self.abs_pos_loss_fn(pos_feat.contiguous().view(-1, F), self.ground_truth_abs_pos.expand(B*T, -1).contiguous().view(-1).to(pos_feat.device)) \
                * self.abs_pos_loss_factor / self.___step
            del pos_feat
            self.___step += 0.001
        else:
            frontend_loss = None


        # B C' T F
        # mask B C C and B T T
        x_time_mask = make_non_pad_mask(ilens, torch.empty(B, T, T).to(x_time_r.device)) * make_non_pad_mask(ilens, torch.empty(B, T, T).to(x_time_r.device), -2)
                
        # BCTF -> BTF
        # print("before", x_time_r.shape, x_time_r)
        x_time_r, x_time_i, x_time_mask = self.beamforming_module(x_time_r, x_time_i, x_time_mask)
        # print("after", x_time_r.shape, x_time_r)

        if getattr(self, "return_mask", False):
            # conj multiply
            x_time_r, x_time_i = (x_time_r*data.real + x_time_i*data.imag).mean(dim=1), (x_time_r*data.imag - x_time_i*data.real).mean(dim=1)
        
        # print(x_time_r.shape)
        # raise
        return ComplexTensor(x_time_r, x_time_i), ilens, frontend_loss


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
        if isinstance(kernel_size, int):
            self.kernel_size_time = kernel_size
            self.kernel_size_channel = kernel_size
        else:
            self.kernel_size_time = kernel_size[0]
            self.kernel_size_channel = kernel_size[1]
        self.dilation = dilation

    def forward(self, x_r, x_i):
        
        # BCTF -> BCTF

        identity_r, identity_i = x_r, x_i
        out = c_F.complex_zero_pad(x_r, x_i, ((self.kernel_size_channel-1)//2, (self.kernel_size_channel-1)//2, self.dilation[0]*(self.kernel_size_time-1), 0))
        del x_r, x_i
        out = self.conv1(*out)
        out = self.bn1(*out) # actually layernorm
        out = self.relu(*out)
        
        out = c_F.complex_zero_pad(out[0], out[1], ((self.kernel_size_channel-1)//2, (self.kernel_size_channel-1)//2, self.dilation[0]*(self.kernel_size_time-1), 0))
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


class BeamformingModule(nn.Module):
    def __init__(
        self, 
        n_blocks: int = 2, 
        n_channel_blocks: int = 2,
        n_time_blocks: int = 2,
        n_head: int = 4,
        in_channel_feat: int = 257,
        in_time_feat: int = 257,
        att_feat: int = 256,
        forward_layer: str = "linear",
        dropout_rate: float = 0,
        reduce_method: str = "mask",
        use_residual: bool = True,
        use_pos_embed: bool = True,
        use_pos_embed_in_beamform_layer: bool = False,
        pos_embed_type: str = "2D",
        module_type: str = "aoliao",
        return_mask: bool = False,
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
            self.before_att_ln = c_nn.ComplexLinear(in_channel_feat, att_feat)
            self.after_att_ln = c_nn.ComplexLinear(att_feat, in_channel_feat)

        if use_pos_embed and not use_pos_embed_in_beamform_layer:
            if pos_embed_type == "2D":
                self.pos_embed = c_nn.ComplexPositionalEncoding2D(att_feat, dropout_rate)
            elif pos_embed_type == "omit_channel":
                self.pos_embed = c_nn.ComplexPositionalEncoding2D(att_feat, dropout_rate, omit_channel=True)
            elif pos_embed_type == "omit_time":
                self.pos_embed = c_nn.ComplexPositionalEncoding2D(att_feat, dropout_rate, omit_time=True)
            else:
                raise ValueError(f"wrong position embed type:{self.pos_embed}")
        else:
            self.pos_embed = c_nn.FakeModule()
        
        self.time_before_norm = c_nn.ComplexLayerNorm(1, affine=False)
        
        if module_type == "aoliao":
            self.att_module = nn.ModuleList()
            for num in range(n_blocks):
                self.att_module.append(
                    BeamformingLayerAoLiAo(
                        n_head=n_head,
                        att_in_feat=att_feat,
                        att_out_feat=att_feat,
                        forward_layer=forward_layer,
                        dropout_rate=dropout_rate,
                        use_pos_embed_in_beamform_layer=use_pos_embed_in_beamform_layer,
                        pos_embed_type=pos_embed_type,
                        )
                )
        elif module_type == "shengdai":
            self.att_module = nn.ModuleList()
            for _ in range(n_channel_blocks-1):
                self.att_module.append(
                    BeamformingLayerShengDai(
                        n_head=n_head,
                        att_in_feat=att_feat,
                        att_out_feat=att_feat,
                        forward_layer=forward_layer,
                        dropout_rate=dropout_rate,
                        transpose_after=False,
                        use_mask=False,
                        use_pos_embed_in_beamform_layer=use_pos_embed_in_beamform_layer,
                        pos_embed_type=pos_embed_type,
                    )
                )
            if n_channel_blocks > 0:
                self.att_module.append(
                    BeamformingLayerShengDai(
                        n_head=n_head,
                        att_in_feat=att_feat,
                        att_out_feat=att_feat,
                        forward_layer=forward_layer,
                        dropout_rate=dropout_rate,
                        transpose_after=True if n_time_blocks > 0 else False,
                        use_mask=False,
                        use_pos_embed_in_beamform_layer=use_pos_embed_in_beamform_layer,
                        pos_embed_type=pos_embed_type,
                    )
                )
            elif n_time_blocks > 0:
                self.att_module.append(
                    c_nn.TransposeModule(
                        transpose_dim=[1,2],
                    )
                )

            for _ in range(n_time_blocks-1):
                self.att_module.append(
                    BeamformingLayerShengDai(
                        n_head=n_head,
                        att_in_feat=att_feat,
                        att_out_feat=att_feat,
                        forward_layer=forward_layer,
                        dropout_rate=dropout_rate,
                        transpose_after=False,
                        use_mask=True,
                        use_pos_embed_in_beamform_layer=use_pos_embed_in_beamform_layer,
                        pos_embed_type=pos_embed_type,
                    )
                )
            if n_time_blocks > 0:
                self.att_module.append(
                    BeamformingLayerShengDai(
                        n_head=n_head,
                        att_in_feat=att_feat,
                        att_out_feat=att_feat,
                        forward_layer=forward_layer,
                        dropout_rate=dropout_rate,
                        transpose_after=True,
                        use_mask=True,
                        use_pos_embed_in_beamform_layer=use_pos_embed_in_beamform_layer,
                        pos_embed_type=pos_embed_type,
                    )
                )
        else:
            raise ValueError(f"Wrong channel module type: {module_type}")
        
        if not return_mask:
            assert reduce_method in ["mask", "mean"], f"reduce_method should be 'mask' or 'mean', but get {reduce_method}"
        self.reduce_method = reduce_method
        self.norm = c_nn.ComplexLayerNorm(1, affine=False)
        self.return_mask = return_mask


    def forward(self, x_channel_r, x_channel_i, x_channel_mask):
        """

        Args:
            x_channel_r, x_channel_i: (batch, channel, time, att_feat)
            x_channel_mask: None or (batch, 1, channel) or (batch, channel, channel)
            x_time_r, x_time_i: (batch, C, time, att_feat)
            x_time_mask: (batch, 1, time) or (batch, time, time)
        return 
            out_r, out_i: (batch, time, att_out_feat)

        """
    
        if hasattr(self,"before_att_ln"):
            out_channel_r, out_channel_i = self.before_att_ln(x_channel_r, x_channel_i)
        else:
            out_channel_r, out_channel_i = x_channel_r, x_channel_i

        # position embedding
        out_channel_r, out_channel_i = self.pos_embed(out_channel_r, out_channel_i)

        for layer in self.att_module:
            out_channel_r, out_channel_i = layer(out_channel_r, out_channel_i, x_channel_mask)
            
        
        # del x_channel_mask

        out_channel_r, out_channel_i = self.norm(out_channel_r, out_channel_i)
        
        if hasattr(self,"after_att_ln"):
            out_channel_r, out_channel_i = self.after_att_ln(out_channel_r, out_channel_i)

        if getattr(self,"return_mask", False):
            # Output: BCTF
            return out_channel_r, out_channel_i, x_channel_mask

        if self.reduce_method == "mask":
            out_channel_r, out_channel_i = out_channel_r*x_channel_r + out_channel_i*x_channel_i, out_channel_r*x_channel_i - out_channel_i*x_channel_r # conj of mask multiply input
        
        del x_channel_r, x_channel_i

        out_channel_r, out_channel_i = out_channel_r.mean(dim=1), out_channel_i.mean(dim=1)
            
        return out_channel_r.contiguous(), out_channel_i.contiguous(), x_channel_mask


class BeamformingLayerAoLiAo(nn.Module):
    def __init__(
        self,
        n_head: int = 4,
        att_in_feat: int = 256,
        att_out_feat: int = 256,
        forward_layer: str = "linear",
        dropout_rate: float = 0,
        use_pos_embed_in_beamform_layer: bool=False,
        pos_embed_type: str= "2D",
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
        assert forward_layer=="linear", f"{forward_layer}"
        
        self.dropout = c_nn.ComplexDropout(dropout_rate)

        self.channel_wise_att_module = c_nn.ComplexMultiHeadedAttention(n_head, att_in_feat, att_in_feat, dropout_rate)
        self.norm1 = c_nn.ComplexLayerNorm(1, affine=False) # we should fix this bug that we have to know the channel even when we don't use "affine"

        self.time_wise_att_module = c_nn.ComplexMultiHeadedAttention(n_head, att_in_feat, att_out_feat, dropout_rate)
        self.norm2 = c_nn.ComplexLayerNorm(1, affine=False)


        self.forward_layer = c_nn.ComplexSequential(
            c_nn.ComplexLinear(att_out_feat, 2048),
            c_nn.ComplexLeakyReLU(inplace=True),
            c_nn.ComplexDropout(dropout_rate),
            c_nn.ComplexLinear(2048, att_out_feat),
            )
        
        self.norm3 = c_nn.ComplexLayerNorm(1, affine=False)

        self.use_pos_embed_in_beamform_layer = use_pos_embed_in_beamform_layer
        if self.use_pos_embed_in_beamform_layer:
            if pos_embed_type == "2D":
                self.pos_embed = c_nn.ComplexPositionalEncoding2D(att_in_feat, dropout_rate)
            elif pos_embed_type == "omit_channel":
                self.pos_embed = c_nn.ComplexPositionalEncoding2D(att_in_feat, dropout_rate, omit_channel=True)
            elif pos_embed_type == "omit_time":
                self.pos_embed = c_nn.ComplexPositionalEncoding2D(att_in_feat, dropout_rate, omit_time=True)
            else:
                raise ValueError("")
        
    def forward(self, x_channel_r, x_channel_i, x_channel_mask):
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

        if hasattr(self, "pos_embed"):
            x_channel_r_pos_enc, x_channel_i_pos_enc = self.pos_embed(x_channel_r, x_channel_i)
            x_channel_r, x_channel_i = self.dropout(*self.att_module(
                [x_channel_r_pos_enc, x_channel_i_pos_enc], 
                [x_channel_r_pos_enc, x_channel_i_pos_enc], 
                [x_channel_r, x_channel_i], 
                None
                ))
            del x_channel_r_pos_enc, x_channel_i_pos_enc
        else:
            x_channel_r, x_channel_i = self.dropout(*self.att_module([x_channel_r, x_channel_i], [x_channel_r, x_channel_i], [x_channel_r, x_channel_i], None))
        
        x_channel_r, x_channel_i = residual_r + x_channel_r, residual_i + x_channel_i

        del residual_r, residual_i

        # Attention in temporal dimension 
        # B C T F -> B T C F
        x_channel_r, x_channel_i = x_channel_r.transpose(1,2).contiguous(), x_channel_i.transpose(1,2).contiguous()
        residual_r, residual_i = x_channel_r, x_channel_i
        

        x_channel_r, x_channel_i = self.norm2(x_channel_r, x_channel_i)
        
        x_channel_r, x_channel_i = self.dropout(*self.time_wise_att_module([x_channel_r, x_channel_i], [x_channel_r, x_channel_i], [x_channel_r, x_channel_i], x_channel_mask))

        x_channel_r, x_channel_i = residual_r + x_channel_r, residual_i + x_channel_i


        # Forward Layer
        residual_r, residual_i = x_channel_r, x_channel_i

        x_channel_r, x_channel_i = self.norm3(x_channel_r, x_channel_i)
        x_channel_r, x_channel_i = self.dropout(*self.forward_layer(x_channel_r, x_channel_i))
        
        x_channel_r, x_channel_i = residual_r + x_channel_r, residual_i + x_channel_i

        return x_channel_r.transpose(1,2).contiguous(), x_channel_i.transpose(1,2).contiguous()


class BeamformingLayerShengDai(nn.Module):
    def __init__(
        self,
        n_head: int = 4,
        att_in_feat: int = 256,
        att_out_feat: int = 256,
        forward_layer: str = "linear",
        dropout_rate: float = 0,
        transpose_after: bool = False,
        use_mask: bool = False,
        use_pos_embed_in_beamform_layer: bool=False,
        pos_embed_type: str= "2D",
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

        if use_pos_embed_in_beamform_layer:
            if pos_embed_type == "2D":
                self.pos_embed = c_nn.ComplexPositionalEncoding2D(att_in_feat, dropout_rate)
            elif pos_embed_type == "omit_channel":
                self.pos_embed = c_nn.ComplexPositionalEncoding2D(att_in_feat, dropout_rate, omit_channel=True)
            elif pos_embed_type == "omit_time":
                self.pos_embed = c_nn.ComplexPositionalEncoding2D(att_in_feat, dropout_rate, omit_time=True)
            else:
                raise ValueError(f"Wrong pos embed type")

        self.dropout = c_nn.ComplexDropout(dropout_rate)

        self.att_module = c_nn.ComplexMultiHeadedAttention(n_head, att_in_feat, att_in_feat, dropout_rate)
        self.norm1 = c_nn.ComplexLayerNorm(1, affine=False) # we should fix this bug that we have to know the channel even when we don't use "affine"

        self.forward_layer = c_nn.ComplexSequential(
            c_nn.ComplexLinear(att_out_feat, 2048),
            c_nn.ComplexLeakyReLU(inplace=True),
            c_nn.ComplexDropout(dropout_rate),
            c_nn.ComplexLinear(2048, att_out_feat),
            )
        
        self.norm2 = c_nn.ComplexLayerNorm(1, affine=False)
        self.transpose_after = transpose_after
        self.use_mask = use_mask
    
        
    def forward(self, x_channel_r, x_channel_i, x_channel_mask):
        """
        :param torch.Tensor x_channel_r, x_channel_i: (batch, channel, time, att_in_feat)
        :param torch.Tensor x_time_r, x_time_i: (batch, time, channel, att_in_feat)
        x_channel_mask: (batch, max_channel, max_channel) or None
        x_time_mask: (batch, max_time_in, max_time_in)
        :return torch.Tensor x_r, x_i: (batch, channel, time, att_out_feat)

        """
        if not self.use_mask:
            x_channel_mask = None
        # Attention
        residual_r, residual_i = x_channel_r, x_channel_i

        x_channel_r, x_channel_i = self.norm1(x_channel_r, x_channel_i)

        if hasattr(self, "pos_embed"):
            x_channel_r_pos_enc, x_channel_i_pos_enc = self.pos_embed(x_channel_r, x_channel_i)
            x_channel_r, x_channel_i = self.dropout(*self.att_module(
                [x_channel_r_pos_enc, x_channel_i_pos_enc], 
                [x_channel_r_pos_enc, x_channel_i_pos_enc], 
                [x_channel_r, x_channel_i], 
                x_channel_mask
                ))
            del x_channel_r_pos_enc, x_channel_i_pos_enc
        else:
            x_channel_r, x_channel_i = self.dropout(*self.att_module([x_channel_r, x_channel_i], [x_channel_r, x_channel_i], [x_channel_r, x_channel_i], x_channel_mask))
        
        x_channel_r, x_channel_i = residual_r + x_channel_r, residual_i + x_channel_i


        # Forward Layer
        residual_r, residual_i = x_channel_r, x_channel_i

        x_channel_r, x_channel_i = self.norm2(x_channel_r, x_channel_i)
        x_channel_r, x_channel_i = self.dropout(*self.forward_layer(x_channel_r, x_channel_i))
        
        x_channel_r, x_channel_i = residual_r + x_channel_r, residual_i + x_channel_i

        if self.transpose_after:
            return x_channel_r.transpose(1,2).contiguous(), x_channel_i.transpose(1,2).contiguous()
        else:
            return x_channel_r, x_channel_i

          
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

class PowerAttentionModule(nn.Module):
     
    def __init__(self, 
                window_len=2, 
                affine=False, 
                softmax_factor=1.0, 
                auto_grad_softmax_factor=False, 
                use_delta_power_att=False, 
                auto_grad_delta_factor=True, 
                delta_factor=3.0, 
                delta_pos=4,
                fft_feat=257, 
                eps=1e-5,
                reduce_method="softmax",
                n_blocks: int = 2, 
                n_channel_blocks: int = 2,
                n_time_blocks: int = 2,
                n_head: int = 4,
                in_channel_feat: int = 257,
                in_time_feat: int = 257,
                att_feat: int = 256,
                forward_layer: str = "linear",
                dropout_rate: float = 0,
                use_residual: bool = True,
                use_pos_embed: bool = True,
                use_pos_embed_in_beamform_layer: bool = False,
                pos_embed_type: str = "2D",
                module_type: str = "aoliao",
                ):
        super().__init__()
        
        if reduce_method == "mean" or reduce_method == "mask":
            reduce_method = "softmax"

        assert reduce_method in ["max", "softmax", "channel_att_max", "double_score", "channel_att_max_v2", "double_score_v2", "double_score_v3"], f'the reduce method of power_att module must be in \
            ["max", "softmax", "channel_att_max", "double_score","channel_att_max_v2", "double_score_v2", "double_score_v3"]'
        # channel_att_max_v2 和 channel_att_max主要区别是max选的一个是channel_att前的max(v1)，v2是做完channel_att再选最大（这个是和seq一样）
        # double_score_v2 和double_score的主要区别是，v2是对channel的权重做了时间上的平均，取了均值。
        # double_score 是对复数的谱做内积，取复数softmax
        # double_score_v2 是对幅值谱做内积，取实数softmax
        # double_score_v3 是对v2再做一个时间上的平均
        print(f"We use {reduce_method} as reduce method")
        self.reduce_method = reduce_method
        
        self.window_len = window_len
        self.affine = affine
        self.use_delta_power_att = use_delta_power_att
        
        
        if self.use_delta_power_att:
            self.delta_pos = delta_pos
            if auto_grad_delta_factor and not self.affine:
                print("we use auto_grad_delta_factor")
                self.delta_factor = torch.nn.Parameter(torch.Tensor([delta_factor]))
            else:
                self.delta_factor = delta_factor

        if auto_grad_softmax_factor and not self.affine:
            print("we use auto_grad_softmax_factor")
            self.softmax_factor = torch.nn.Parameter(torch.Tensor([softmax_factor]))
        else:
            self.softmax_factor = softmax_factor

        if self.affine:
            print("we use affine in power attention module")
            print("Warning: When using 'affine' in power attention module, the settings of auto_grad_softmax_factor and auto_grad_delta_factor will be switch to False")
            # self.fft_ln = c_nn.ComplexLinear(fft_feat, fft_feat)
            self.ln_left = nn.Linear(fft_feat, fft_feat)
            self.ln_right = nn.Linear(fft_feat, fft_feat)
        if self.reduce_method == "channel_att_max" or self.reduce_method == "channel_att_max_v2":
            self.channel_att_layer  = BeamformingModule(
                        n_blocks=n_blocks,
                        n_channel_blocks=n_channel_blocks,
                        n_time_blocks=n_time_blocks,
                        n_head=n_head,
                        in_channel_feat=in_channel_feat,
                        in_time_feat=in_time_feat,
                        att_feat=att_feat,
                        forward_layer=forward_layer,
                        dropout_rate=dropout_rate,
                        use_residual=use_residual,
                        use_pos_embed = use_pos_embed,
                        pos_embed_type = pos_embed_type,
                        module_type = module_type,
                        return_mask=True,
                        use_pos_embed_in_beamform_layer=use_pos_embed_in_beamform_layer,
                    )
        if self.reduce_method == "double_score":
            self.channel_score_ln = c_nn.ComplexLinear(fft_feat,fft_feat)
            self.softmax = c_nn.ComplexSoftmax(dim=-1)
        if self.reduce_method == "double_score_v2" or self.reduce_method == "double_score_v3":
            self.channel_score_ln = nn.Linear(fft_feat,fft_feat)
            self.softmax = nn.Softmax(dim=-1)

        self.eps = eps

    ################################## Time domain att 
    # def _affine_forward(self, x_r, x_i, mask=None):
        
    #     B, C, T, F = x_r.shape
        
    #     # FFT and take the real part
    #     time_x, __ = self.fft_ln(x_r, x_i)

    #     # norm to power eq 1
    #     time_x = time_x / (time_x.pow(2).sum(dim=-1).unsqueeze(-1).sqrt() + self.eps)
        
    #     # add mask
    #     if isinstance(mask, torch.Tensor):
    #         # BTT -> BT -> B -> B1
    #         batch_len = mask[:, 0].sum(-1).unsqueeze(-1)
    #         # BTT -> B1T -> B1T1 -> BCTF
    #         mask = mask[:,0:1].unsqueeze(-1).expand_as(x_r).eq(0)
    #         time_x = time_x.masked_fill(mask, 0.)

    #         power_att = torch.zeros(B,C).to(time_x.device)

    #         for step in range(1, self.window_len):
    #             end_index = -self.window_len+1+step if -self.window_len+1+step < 0 else None
    #             power_att += (0.5**(step-1)) * (amp_x[:, :, :-self.window_len+1] * amp_x[:, :, step:end_index]).sum(dim=-1).abs().sum(dim=-1) / batch_len
    #     else:
    #         power_att = torch.zeros(B,C).to(time_x.device)

    #         for step in range(1, self.window_len):
    #             end_index = -self.window_len+1+step if -self.window_len+1+step < 0 else None
    #             power_att += (0.5**(step-1)) * (amp_x[:, :, :-self.window_len+1] * amp_x[:, :, step:end_index]).sum(dim=-1).abs().mean(dim=-1)

    #     power_att = power_att * self.softmax_factor
    #     # BC -> softmax() -> BC11
    #     power_att = torch.nn.functional.softmax(power_att, dim=-1).unsqueeze(-1).unsqueeze(-1)

    #     # BCTF -> BCTF -> BTF
    #     return (power_att*x_r).sum(dim=1), (power_att*x_i).sum(dim=1), mask
    
    def _affine_forward(self, x_r, x_i, mask=None):
        
        B, C, T, F = x_r.shape

        # power BCTF
        # with torch.no_grad():
        #     power_x = x_r.pow(2) + x_i.pow(2)
        #     # norm
        #     power_x = power_x / (power_x.sum(dim=-1).unsqueeze(-1) + self.eps)
        #     # amplitude BCTF
        #     amp_x = (power_x + self.eps).sqrt()
        #     del power_x
        #     # add mask
            # if isinstance(mask, torch.Tensor):
            #     # BTT -> B1T -> B1T1 -> BCTF
            #     mask = mask[:,0:1].unsqueeze(-1).expand_as(x_r).eq(0)
            #     amp_x = amp_x.masked_fill(mask, 0.)

            # power_att = torch.zeros(B,C).to(amp_x.device)

            # if self.affine:
            #     for step in range(1, self.window_len):
            #         end_index = -self.window_len+1+step if -self.window_len+1+step < 0 else None
            #         power_att += (0.5**(step-1)) * (self.ln_left(amp_x[:, :, :-self.window_len+1]) * self.ln_right(amp_x[:, :, step:end_index])).sum(dim=-1).mean(dim=-1)
            # else:
            #     for step in range(1, self.window_len):
            #         end_index = -self.window_len+1+step if -self.window_len+1+step < 0 else None
            #         power_att += (0.5**(step-1)) * (amp_x[:, :, :-self.window_len+1] * amp_x[:, :, step:end_index]).sum(dim=-1).mean(dim=-1)

        power_x = x_r.pow(2) + x_i.pow(2)
        # norm
        power_x = power_x / (power_x.sum(dim=-1).unsqueeze(-1) + self.eps)
        # amplitude BCTF
        amp_x = (power_x + self.eps).sqrt()
        del power_x

        ### 和之前兼容的版本#########################
        # if isinstance(mask, torch.Tensor):
        #     # BTT -> B1T -> B1T1 -> BCTF
        #     mask = mask[:,0:1].unsqueeze(-1).expand_as(x_r).eq(0)
        #     amp_x = amp_x.masked_fill(mask, 0.)

        # power_att = torch.zeros(B,C).to(amp_x.device)

        # for step in range(1, self.window_len):
        #     end_index = -self.window_len+1+step if -self.window_len+1+step < 0 else None
        #     power_att += (0.5**(step-1)) * (amp_x[:, :, :-self.window_len+1] * amp_x[:, :, step:end_index]).sum(dim=-1).mean(dim=-1)
        ############################################

        ### 正确的 2.23##################################
        # add mask
        if isinstance(mask, torch.Tensor):
            # BTT -> BT -> B -> B1
            batch_len = mask[:, 0].sum(-1).unsqueeze(-1)
            # BTT -> B1T -> B1T1 -> BCTF
            mask = mask[:,0:1].unsqueeze(-1).expand_as(x_r).eq(0)
            amp_x = amp_x.masked_fill(mask, 0.)

            power_att = torch.zeros(B,C).to(amp_x.device)

            for step in range(1, self.window_len):
                end_index = -self.window_len+1+step if -self.window_len+1+step < 0 else None
                power_att += (0.5**(step-1)) * (self.ln_left(amp_x[:, :, :-self.window_len+1]) * self.ln_right(amp_x[:, :, step:end_index])).sum(dim=-1).sum(dim=-1) / batch_len
        else:
            power_att = torch.zeros(B,C).to(amp_x.device)

            for step in range(1, self.window_len):
                end_index = -self.window_len+1+step if -self.window_len+1+step < 0 else None
                power_att += (0.5**(step-1)) * (self.ln_left(amp_x[:, :, :-self.window_len+1]) * self.ln_right(amp_x[:, :, step:end_index])).sum(dim=-1).mean(dim=-1)
        ##############################################

        power_att = power_att * self.softmax_factor
        # BC -> softmax() -> BC11
        power_att = torch.nn.functional.softmax(power_att, dim=-1).unsqueeze(-1).unsqueeze(-1)

        # BCTF -> BCTF -> BTF
        return (power_att*x_r).sum(dim=1), (power_att*x_i).sum(dim=1), mask


    def _no_affine_forward(self, x_r, x_i, mask=None):
        
        B, C, T, F = x_r.shape

        # power BCTF
        # with torch.no_grad():
        #     power_x = x_r.pow(2) + x_i.pow(2)
        #     # norm
        #     power_x = power_x / (power_x.sum(dim=-1).unsqueeze(-1) + self.eps)
        #     # amplitude BCTF
        #     amp_x = (power_x + self.eps).sqrt()
        #     del power_x
        #     # add mask
            # if isinstance(mask, torch.Tensor):
            #     # BTT -> B1T -> B1T1 -> BCTF
            #     mask = mask[:,0:1].unsqueeze(-1).expand_as(x_r).eq(0)
            #     amp_x = amp_x.masked_fill(mask, 0.)

            # power_att = torch.zeros(B,C).to(amp_x.device)

            # if self.affine:
            #     for step in range(1, self.window_len):
            #         end_index = -self.window_len+1+step if -self.window_len+1+step < 0 else None
            #         power_att += (0.5**(step-1)) * (self.ln_left(amp_x[:, :, :-self.window_len+1]) * self.ln_right(amp_x[:, :, step:end_index])).sum(dim=-1).mean(dim=-1)
            # else:
            #     for step in range(1, self.window_len):
            #         end_index = -self.window_len+1+step if -self.window_len+1+step < 0 else None
            #         power_att += (0.5**(step-1)) * (amp_x[:, :, :-self.window_len+1] * amp_x[:, :, step:end_index]).sum(dim=-1).mean(dim=-1)

        power_x = x_r.pow(2) + x_i.pow(2)
        # norm
        power_x = power_x / (power_x.sum(dim=-1).unsqueeze(-1) + self.eps)
        # amplitude BCTF
        amp_x = (power_x + self.eps).sqrt()
        del power_x

        ### 和之前兼容的版本#########################
        if isinstance(mask, torch.Tensor):
            # BTT -> B1T -> B1T1 -> BCTF
            mask = mask[:,0:1].unsqueeze(-1).expand_as(x_r).eq(0)
            amp_x = amp_x.masked_fill(mask, 0.)

        power_att = torch.zeros(B,C).to(amp_x.device)

        for step in range(1, self.window_len):
            end_index = -self.window_len+1+step if -self.window_len+1+step < 0 else None
            power_att += (0.5**(step-1)) * (amp_x[:, :, :-self.window_len+1] * amp_x[:, :, step:end_index]).sum(dim=-1).mean(dim=-1)
        ############################################

        ### 正确的 2.23##################################
        # add mask
        # if isinstance(mask, torch.Tensor):
        #     # BTT -> BT -> B -> B1
        #     batch_len = mask[:, 0].sum(-1).unsqueeze(-1)
        #     # BTT -> B1T -> B1T1 -> BCTF
        #     mask = mask[:,0:1].unsqueeze(-1).expand_as(x_r).eq(0)
        #     amp_x = amp_x.masked_fill(mask, 0.)

        #     power_att = torch.zeros(B,C).to(amp_x.device)

        #     for step in range(1, self.window_len):
        #         end_index = -self.window_len+1+step if -self.window_len+1+step < 0 else None
        #         power_att += (0.5**(step-1)) * (amp_x[:, :, :-self.window_len+1] * amp_x[:, :, step:end_index]).sum(dim=-1).sum(dim=-1) / batch_len
        # else:
        #     power_att = torch.zeros(B,C).to(amp_x.device)

        #     for step in range(1, self.window_len):
        #         end_index = -self.window_len+1+step if -self.window_len+1+step < 0 else None
        #         power_att += (0.5**(step-1)) * (amp_x[:, :, :-self.window_len+1] * amp_x[:, :, step:end_index]).sum(dim=-1).mean(dim=-1)
        ##############################################

        power_att = power_att * self.softmax_factor
        # BC -> softmax() -> BC11
        power_att = torch.nn.functional.softmax(power_att, dim=-1).unsqueeze(-1).unsqueeze(-1)

        # BCTF -> BCTF -> BTF
        return (power_att*x_r).sum(dim=1), (power_att*x_i).sum(dim=1), mask

    def _delta_power_forward(self, x_r, x_i, mask=None):
        
        B, C, T, F = x_r.shape

        if self.reduce_method == "channel_att_max_v2":
            x_r, x_i, _ = self.channel_att_layer(x_r, x_i, None)

        power_x = x_r.pow(2) + x_i.pow(2)
        # norm
        power_x = power_x / (power_x.sum(dim=-1).unsqueeze(-1) + self.eps)
        # amplitude BCTF
        amp_x = (power_x + self.eps).sqrt()
        del power_x

        ### 正确的 2.23##################################
        # add mask
        if isinstance(mask, torch.Tensor):
            # BTT -> BT -> B -> B1
            batch_len = mask[:, 0].sum(-1).unsqueeze(-1)
            batch_len = batch_len - (self.delta_pos - 1)
            end_index = T - (self.delta_pos - 1)
            # BTT -> B1T -> B1T1 -> BCTF
            mask = mask[:,0:1].unsqueeze(-1).expand_as(x_r).eq(0)
            amp_x = amp_x.masked_fill(mask, 0.)

            if self.affine:
                power_att = (self.ln_left(amp_x[:, :, : end_index]) * amp_x[:, :, 1: end_index + 1]).sum(dim=-1).sum(dim=-1) / batch_len \
                    - self.delta_factor * (amp_x[:, :, : end_index]* self.ln_right(amp_x[:, :, self.delta_pos-1:end_index+self.delta_pos-1])).sum(dim=-1).sum(dim=-1) / batch_len
            else:
                power_att = (amp_x[:, :, : end_index] * amp_x[:, :, 1: end_index + 1]).sum(dim=-1).sum(dim=-1) / batch_len \
                    - self.delta_factor * (amp_x[:, :, : end_index] * amp_x[:, :, self.delta_pos-1:end_index+self.delta_pos-1]).sum(dim=-1).sum(dim=-1) / batch_len

            # for step in range(1, self.window_len):
            #     end_index = -self.window_len+1+step if -self.window_len+1+step < 0 else None
            #     power_att += (amp_x[:, :, :-self.window_len+1] * amp_x[:, :, step:end_index]).sum(dim=-1).sum(dim=-1) / batch_len
        else:
            
            batch_len = T - (self.delta_pos - 1)

            if self.affine:
                power_att = (self.ln_left(amp_x[:, :, : batch_len]) * self.ln_right(amp_x[:, :, 1: batch_len + 1])).sum(dim=-1).sum(dim=-1) / batch_len \
                    - self.delta_factor * (self.ln_left(amp_x[:, :, : batch_len]) * self.ln_right(amp_x[:, :, self.delta_pos-1:batch_len+self.delta_pos-1])).sum(dim=-1).sum(dim=-1) / batch_len
            else:
                power_att = (amp_x[:, :, : batch_len] * amp_x[:, :, 1: batch_len + 1]).sum(dim=-1).sum(dim=-1) / batch_len \
                    - self.delta_factor * (amp_x[:, :, : batch_len] * amp_x[:, :, self.delta_pos-1:batch_len+self.delta_pos-1]).sum(dim=-1).sum(dim=-1) / batch_len

            # power_att = torch.zeros(B,C).to(amp_x.device)

            # for step in range(1, self.window_len):
            #     end_index = -self.window_len+1+step if -self.window_len+1+step < 0 else None
            #     power_att += (0.5**(step-1)) * (amp_x[:, :, :-self.window_len+1] * amp_x[:, :, step:end_index]).sum(dim=-1).mean(dim=-1)
        ##############################################

        ## time domain scores
        power_att = power_att * self.softmax_factor
        if self.reduce_method == "max" or self.reduce_method == "channel_att_max" or self.reduce_method == "channel_att_max_v2":
            # BC -> BC -> BC11
            max_mask = torch.eq(power_att, torch.max(power_att, dim=1, keepdim=True)[0]).unsqueeze(-1).unsqueeze(-1).float()
            if self.reduce_method == "channel_att_max":
                x_r, x_i, _ = self.channel_att_layer(x_r, x_i, None)

            return (max_mask*x_r).sum(dim=1), (max_mask*x_i).sum(dim=1), mask

        # BC -> softmax() -> BC11
        power_att = torch.nn.functional.softmax(power_att, dim=-1).unsqueeze(-1).unsqueeze(-1)


        if self.reduce_method == "softmax":
            # BCTF -> BCTF -> BTF
            return (power_att*x_r).sum(dim=1), (power_att*x_i).sum(dim=1), mask
        elif self.reduce_method == "double_score":
            left_r, left_i = self.channel_score_ln(x_r, x_i)
            # B C T F -> B T C F
            left_r, left_i = left_r.transpose(1,2), left_i.transpose(1,2)
            # B C T F -> B T F C (conj)
            right_r, right_i = x_r.permute([0,2,3,1]), -x_i.permute([0,2,3,1])

            # B T C C
            channel_score_r, channel_score_i = \
                torch.matmul(left_r, right_r)-torch.matmul(left_i, right_i), \
                torch.matmul(left_r, right_i) + torch.matmul(left_i, right_r)
            del left_r, left_i, right_r, right_i
            # 这步有问题。。softmax和下一步mean的维度一样了，double_score有问题
            channel_score_r, channel_score_i = self.softmax(channel_score_r, channel_score_i)
            # B T C C -> B T C -> B C T -> B C T 1
            channel_score_r, channel_score_i = channel_score_r.mean(dim=-1).transpose(1,2).unsqueeze(-1), channel_score_i.mean(dim=-1).transpose(1,2).unsqueeze(-1)
            
            return (power_att*(channel_score_r*x_r - channel_score_i*x_i)).sum(dim=1), (power_att*(channel_score_r*x_i + channel_score_i*x_r)).sum(dim=1), mask
        elif self.reduce_method == "double_score_v2" or self.reduce_method == "double_score_v3":
            # BCTF -> BTCF
            left = self.channel_score_ln(amp_x).transpose(1,2)
            # BCTF -> BTFC
            right = amp_x.permute([0,2,3,1])
            # B T C C
            channel_score = torch.matmul(left, right)
            # B T C C -> B T 1 C -> B T 1 C -> B C T 1
            channel_score = self.softmax(channel_score.mean(dim=-2, keepdim=True)).permute([0,3,1,2])
            if self.reduce_method == "double_score_v3":
                channel_score = channel_score.mean(dim=-2, keepdim=True)
            return (power_att * channel_score * x_r).sum(dim=1), (power_att * channel_score * x_i).sum(dim=1), mask


        

    def forward(self, x_r, x_i, mask=None):
        '''
            x_r, x_i: torch.Tensor B C' T F
            mask: BTT (make_non_pad_mask)
            return:
                x_r, x_i torch.Tensor B T F
        '''
        if self.use_delta_power_att:
            return self._delta_power_forward(x_r, x_i, mask)
    
        if self.affine:
            return self._affine_forward(x_r, x_i, mask)
        else:
            return self._no_affine_forward(x_r, x_i, mask)


class CNN_Front_Attention_Mean_Universal(nn.Module):
    '''
        Use CNN as the front and use power
        attention or self-attention to get 
        mono output

        Output enhanced STFT: B x T x F
    '''

    def __init__(
        self, 
        conv_layer_list: List[int] = [4, 8, 16, 32], 
        inplane=1, 
        use_dilation: bool = True,
        n_att_head: int = 4,
        n_att_blocks: int = 2,
        n_channel_att_blocks: int = 2,
        n_time_att_blocks: int = 2,
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
        use_pos_embed_in_beamform_layer: bool = False,
        pos_embed_type: str = "2D",
        beamformer_type: str = "shengdai",
        beamformer_return_mask: bool = False,
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

        if beamformer_return_mask:
            print("We multiply the mask with the orig input")

        self.return_mask = beamformer_return_mask

        self.beamforming_module = BeamformingModule(
            n_blocks=n_att_blocks,
            n_channel_blocks=n_channel_att_blocks,
            n_time_blocks=n_time_att_blocks,
            n_head=n_att_head,
            in_channel_feat=fft_feat,
            in_time_feat=fft_feat,
            att_feat=att_feat,
            forward_layer="linear",
            dropout_rate=dropout_rate,
            reduce_method=reduce_method,
            use_residual=use_residual,
            use_pos_embed = use_pos_embed,
            pos_embed_type = pos_embed_type,
            module_type = beamformer_type,
            return_mask=self.return_mask,
            use_pos_embed_in_beamform_layer=use_pos_embed_in_beamform_layer,
        )
        if use_pos_embed_in_beamform_layer:
            print("We use pos embed on q and k not v")

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
        x_time_r, x_time_i = self.data_norm(data.real, data.imag)
        
        if not getattr(self, "return_mask", False):
            del data

        B, C, T, F = x_time_r.shape 

        x_time_r, x_time_i = x_time_r.contiguous().view(B*C, 1, T, F), x_time_i.contiguous().view(B*C, 1, T, F)

        # Beamformer / ResNet: B*C 1 T F -> B*C C' T F
        for conv in self.conv_layers:
            x_time_r, x_time_i = conv(x_time_r, x_time_i)

        # B*C C' T F -> B*C 1 T F 
        if hasattr(self, "downconv"):
            if getattr(self, "downconv_type", False) == "stack_linear":
               # B C' T F -> B 1 T C'*F
               x_time_r, x_time_i = x_time_r.transpose(1, 2).contiguous().view(B*C, 1, T, -1), x_time_i.transpose(1,2).contiguous().view(B*C, 1, T, -1)
            
            x_time_r, x_time_i = self.downconv(x_time_r, x_time_i)
        
        # B*C 1 T F -> B*C T F
        if hasattr(self, "sub_sample_layer"):
            x_time_r, x_time_i, ilens = self.sub_sample_layer(x_time_r, x_time_i, ilens)
            T = x_time_r.shape[1]
        
        # B*C 1 T F or B*C T F -> BCTF
        x_time_r, x_time_i = x_time_r.contiguous().view(B, C, -1, F), x_time_i.contiguous().view(B, C, -1, F)
    
        # mask B C C and B T T
        x_time_mask = make_non_pad_mask(ilens, torch.empty(B, T, T).to(x_time_r.device)) * make_non_pad_mask(ilens, torch.empty(B, T, T).to(x_time_r.device), -2)
            
        # BCTF -> BTF
        x_time_r, x_time_i = self.beamforming_module(x_time_r, x_time_i, x_time_mask)

        if self.return_mask:
            # conj multiply
            x_time_r, x_time_i = (x_time_r*data.real + x_time_i*data.imag).mean(dim=1), (x_time_r*data.imag - x_time_i*data.real).mean(dim=1)

        return ComplexTensor(x_time_r, x_time_i), ilens, None


class MeanModule(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x_r, x_i, x_mask=None):
        return x_r.mean(self.dim), x_i.mean(self.dim), x_mask


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

