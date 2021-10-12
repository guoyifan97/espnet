from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy
import torch
import torch.nn as nn
from torch_complex.tensor import ComplexTensor

# from espnet.nets.pytorch_backend.frontends.complex_beamformer import Complex_Beamformer


class Frontend(nn.Module):
    def __init__(
        self,
        args,
        idim: int,
        # WPE options
        use_wpe: bool = False,
        wtype: str = "blstmp",
        wlayers: int = 3,
        wunits: int = 300,
        wprojs: int = 320,
        wdropout_rate: float = 0.0,
        taps: int = 5,
        delay: int = 3,
        use_dnn_mask_for_wpe: bool = True,
        # Beamformer options
        use_beamformer: bool = False,
        use_beamformer_guo: bool = False,
        use_complex_beamformer: bool = False,
        no_reverb = False,
        btype: str = "blstmp",
        blayers: int = 3,
        bunits: int = 300,
        bprojs: int = 320,
        bnmask: int = 2,
        badim: int = 320,
        ref_channel: int = -1,
        bdropout_rate=0.0,
        cb_conv_layer_list: List[int] = 8*[257], 
        cb_inplane: int = 257, 
        cb_num_dereverb_frames: int = 90, 
        cb_num_dereverb_blocks: int = 3, 
        cb_use_dereverb_loss: bool = False, 
        cb_use_clean_loss: bool = False, 
        cb_ratio_reverb: float = 0.4,
        cb_down_sample: bool = False,
        cb_down_sample_layer: List[Tuple[str]] = [(16, 7, 2), (64, 5, 2)],
    ):
        super().__init__()

        self.use_beamformer = use_beamformer
        self.use_beamformer_guo = use_beamformer_guo
        self.use_complex_beamformer = use_complex_beamformer
        self.use_universal_beamformer = getattr(args, "use_universal_beamformer", False)
        print(args)
        raise
        self.random_pick_channel = getattr(args, "random_pick_channel", 0)
        if self.random_pick_channel != 0:
            print(f"We shall pick {self.random_pick_channel} channel to train nn")
        self.use_wpe = use_wpe
        self.use_dnn_mask_for_wpe = use_dnn_mask_for_wpe
        # use frontend for all the data,
        # e.g. in the case of multi-speaker speech separation
        self.use_frontend_for_all = bnmask > 2
        self.noise_to_backend = getattr(args,"noisy_to_backend", False)
        if self.noise_to_backend:
            print("We will send the noisy data directly to backend.")

        if self.use_wpe:
            from espnet.nets.pytorch_backend.frontends.dnn_wpe import DNN_WPE
            if self.use_dnn_mask_for_wpe:
                # Use DNN for power estimation
                # (Not observed significant gains)
                iterations = 1
            else:
                # Performing as conventional WPE, without DNN Estimator
                iterations = 5

            self.wpe = DNN_WPE(
                wtype=wtype,
                widim=idim,
                wunits=wunits,
                wprojs=wprojs,
                wlayers=wlayers,
                taps=taps,
                delay=delay,
                dropout_rate=wdropout_rate,
                iterations=iterations,
                use_dnn_mask=use_dnn_mask_for_wpe,
            )
        else:
            self.wpe = None

        if self.use_beamformer:
            if self.use_beamformer_guo:
                from espnet.nets.pytorch_backend.frontends.dnn_beamformer_v2 import DNN_Beamformer_V2
                self.beamformer = DNN_Beamformer_V2(
                    btype=btype,
                    bidim=idim,
                    bunits=bunits,
                    bprojs=bprojs,
                    blayers=blayers,
                    bnmask=bnmask,
                    dropout_rate=bdropout_rate,
                    badim=badim,
                    ref_channel=ref_channel,
            )
            elif self.use_complex_beamformer:
                if no_reverb:
                    print("NOTE: We use frontend model process reverb and noise all together.")
                    if getattr(args,"complex_counterpart",False):
                        from espnet.nets.pytorch_backend.frontends.complex_beamformer_no_reverb_contrast import Complex_Beamformer as Complex_Beamformer_counter
                        self.beamformer = Complex_Beamformer_counter(
                            conv_layer_list=cb_conv_layer_list,
                            inplane=cb_inplane,
                            num_dereverb_frames=cb_num_dereverb_frames, 
                            num_dereverb_blocks=cb_num_dereverb_blocks, 
                            use_dereverb_loss=cb_use_dereverb_loss, 
                            use_clean_loss=cb_use_clean_loss, 
                            ratio_reverb=cb_ratio_reverb,
                            down_sample=cb_down_sample,
                            down_sample_layer=cb_down_sample_layer,
                        )
                    elif getattr(args,"cb_dilation", False):
                        from espnet.nets.pytorch_backend.frontends.complex_beamformer_no_reverb_dilation import Complex_Beamformer as Complex_Beamformer_dilation
                        self.beamformer = Complex_Beamformer_dilation(
                            conv_layer_list=cb_conv_layer_list,
                            inplane=cb_inplane,
                            num_dereverb_frames=cb_num_dereverb_frames, 
                            num_dereverb_blocks=cb_num_dereverb_blocks, 
                            use_dereverb_loss=cb_use_dereverb_loss, 
                            use_clean_loss=cb_use_clean_loss, 
                            ratio_reverb=cb_ratio_reverb,
                            down_sample=cb_down_sample,
                            down_sample_layer=cb_down_sample_layer,
                            use_dilation=args.cb_dilation_replace_stride,
                        )
                    else:
                        from espnet.nets.pytorch_backend.frontends.complex_beamformer_no_reverb import Complex_Beamformer as Complex_Beamformer_no_reverb
                        self.beamformer = Complex_Beamformer_no_reverb(
                            conv_layer_list=cb_conv_layer_list,
                            conv_layer_dilation=eval(getattr(args,"cb_conv_layer_dilation","8*[1]")),
                            inplane=cb_inplane,
                            num_dereverb_frames=cb_num_dereverb_frames, 
                            num_dereverb_blocks=cb_num_dereverb_blocks, 
                            use_dereverb_loss=cb_use_dereverb_loss, 
                            use_clean_loss=cb_use_clean_loss, 
                            ratio_reverb=cb_ratio_reverb,
                            down_sample=cb_down_sample,
                            down_sample_layer=cb_down_sample_layer,
                        )
                else:
                    from espnet.nets.pytorch_backend.frontends.complex_beamformer_v2 import Complex_Beamformer
                    self.beamformer = Complex_Beamformer(
                        conv_layer_list=cb_conv_layer_list,
                        inplane=cb_inplane,
                        num_dereverb_frames=cb_num_dereverb_frames, 
                        num_dereverb_blocks=cb_num_dereverb_blocks, 
                        use_dereverb_loss=cb_use_dereverb_loss, 
                        use_clean_loss=cb_use_clean_loss, 
                        ratio_reverb=cb_ratio_reverb,
                    )
            elif getattr(args, "use_universal_beamformer", False):
                if getattr(args, "use_universal_beamformer_time_attention", False):
                    from espnet.nets.pytorch_backend.frontends.universal_frontend_time_attention import Universal_Frontend_Time_Attention
                    self.beamformer = Universal_Frontend_Time_Attention(
                        conv_layer_list=cb_conv_layer_list,
                        inplane=cb_inplane,
                        use_dilation=args.cb_dilation_replace_stride,
                        n_att_head=args.uf_n_att_head,
                        n_att_blocks=args.uf_n_att_blocks,
                        fft_feat=args.uf_n_fft_feat,
                        att_feat=args.uf_n_att_feat,
                        dropout_rate=args.dropout_rate,
                        reduce_method=getattr(args, "uf_reduce_method", "mask"),
                        use_residual=getattr(args, "uf_use_residual", True),
                        use_sub_sampling=getattr(args, "uf_use_subsampling", False),
                        downconv_type=getattr(args, "uf_downconv_type", "conv1d"),
                    )
                elif getattr(args, "use_universal_beamformer_src_time_attention_pos_enc", False):
                    from espnet.nets.pytorch_backend.frontends.universal_frontend_src_time_attention_pos_enc import Universal_Frontend_Src_Time_Attention_Pos_Enc
                    self.beamformer = Universal_Frontend_Src_Time_Attention_Pos_Enc(
                        conv_layer_list=cb_conv_layer_list,
                        inplane=cb_inplane,
                        use_dilation=args.cb_dilation_replace_stride,
                        n_att_head=args.uf_n_att_head,
                        n_att_blocks=args.uf_n_att_blocks,
                        fft_feat=args.uf_n_fft_feat,
                        att_feat=args.uf_n_att_feat,
                        dropout_rate=args.dropout_rate,
                        reduce_method=getattr(args, "uf_reduce_method", "mask"),
                        use_residual=getattr(args, "uf_use_residual", True),
                        use_sub_sampling=getattr(args, "uf_use_subsampling", False),
                        downconv_type=getattr(args, "uf_downconv_type", "conv1d"),
                        use_time_high_dim_as_v=getattr(args, "uf_use_time_high_dim_as_v", False),
                        use_pos_embed=getattr(args, "uf_use_pos_embed", True),
                        pos_embed_type=getattr(args, "uf_pos_embed_type", "2D"),
                    )
                elif getattr(args, "use_universal_beamformer_time_attention_pos_enc", False):
                    from espnet.nets.pytorch_backend.frontends.universal_frontend_time_attention_pos_encoding import Universal_Frontend_Time_Attention_Pos_Enc
                    self.beamformer = Universal_Frontend_Time_Attention_Pos_Enc(
                        conv_layer_list=cb_conv_layer_list,
                        inplane=cb_inplane,
                        use_dilation=args.cb_dilation_replace_stride,
                        n_att_head=args.uf_n_att_head,
                        n_att_blocks=args.uf_n_att_blocks,
                        n_channel_att_blocks=args.uf_n_channel_att_blocks,
                        n_time_att_blocks=args.uf_n_time_att_blocks,
                        fft_feat=args.uf_n_fft_feat,
                        att_feat=args.uf_n_att_feat,
                        dropout_rate=args.dropout_rate,
                        reduce_method=getattr(args, "uf_reduce_method", "mask"),
                        use_residual=getattr(args, "uf_use_residual", True),
                        use_sub_sampling=getattr(args, "uf_use_subsampling", False),
                        downconv_type=getattr(args, "uf_downconv_type", "conv1d"),
                        use_time_high_dim_as_v=getattr(args, "uf_use_time_high_dim_as_v", False),
                        use_pos_embed=getattr(args, "uf_use_pos_embed", True),
                        pos_embed_type=getattr(args, "uf_pos_embed_type", "2D"),
                        beamformer_type=getattr(args, "uf_beamformer_type", "shengdai"),
                        beamformer_return_mask=getattr(args, "uf_beamformer_return_mask", False),
                    )
                elif getattr(args, "use_cnn_front_attention_mean", False):
                    from espnet.nets.pytorch_backend.frontends.cnn_front_attention_mean import CNN_Front_Attention_Mean
                    self.beamformer = CNN_Front_Attention_Mean(
                        conv_layer_list=cb_conv_layer_list,
                        inplane=cb_inplane,
                        use_dilation=args.cb_dilation_replace_stride,
                        n_att_head=args.uf_n_att_head,
                        n_att_blocks=args.uf_n_att_blocks,
                        n_channel_att_blocks=args.uf_n_channel_att_blocks,
                        n_time_att_blocks=args.uf_n_time_att_blocks,
                        fft_feat=args.uf_n_fft_feat,
                        att_feat=args.uf_n_att_feat,
                        dropout_rate=args.dropout_rate,
                        reduce_method=getattr(args, "uf_reduce_method", "mask"),
                        use_residual=getattr(args, "uf_use_residual", True),
                        use_sub_sampling=getattr(args, "uf_use_subsampling", False),
                        downconv_type=getattr(args, "uf_downconv_type", "conv1d"),
                        use_time_high_dim_as_v=getattr(args, "uf_use_time_high_dim_as_v", False),
                        use_pos_embed=getattr(args, "uf_use_pos_embed", True),
                        pos_embed_type=getattr(args, "uf_pos_embed_type", "2D"),
                        beamformer_type=getattr(args, "uf_beamformer_type", "shengdai"),
                        beamformer_return_mask=getattr(args, "uf_beamformer_return_mask", False),
                        use_abs_pos_pred=getattr(args, "uf_use_abs_pos_pred", True),
                        num_abs_pos_extracted=getattr(args, "uf_num_abs_pos_extracted", 1),
                        attention_method=getattr(args, "uf_attention_method", "channel_wise_att"),
                        power_att_window_len=getattr(args, "uf_power_att_window_len", 2),
                        power_att_affine=getattr(args, "uf_power_att_affine", False),
                        power_att_softmax_factor=getattr(args, "uf_power_att_softmax_factor", 1.0),
                        power_att_auto_grad_softmax_factor=getattr(args, "uf_power_att_auto_grad_softmax_factor", False),
                        power_att_use_delta_power_att=getattr(args, "uf_power_att_use_delta_power_att", False), 
                        power_att_auto_grad_delta_factor=getattr(args, "uf_power_att_auto_grad_delta_factor", True),
                        power_att_delta_factor=getattr(args, "uf_power_att_delta_factor", 3.0),
                        power_att_delta_pos=getattr(args, "uf_power_att_delta_pos", 4),
                        abs_pos_loss_factor=getattr(args, "uf_abs_pos_loss_factor", 50.),
                        use_pos_embed_in_beamform_layer=getattr(args, "uf_use_pos_embed_in_beamform_layer", False),
                        conv_layer_dilation=eval(getattr(args, "cb_conv_layer_dilation", "[1]*8")),
                    )
                elif getattr(args, "use_cnn_front_attention_mean_no_padding", False):
                    from espnet.nets.pytorch_backend.frontends.cnn_front_attention_mean_no_padding import CNN_Front_Attention_Mean
                    self.beamformer = CNN_Front_Attention_Mean(
                        conv_layer_list=cb_conv_layer_list,
                        inplane=cb_inplane,
                        use_dilation=args.cb_dilation_replace_stride,
                        n_att_head=args.uf_n_att_head,
                        n_att_blocks=args.uf_n_att_blocks,
                        n_channel_att_blocks=args.uf_n_channel_att_blocks,
                        n_time_att_blocks=args.uf_n_time_att_blocks,
                        fft_feat=args.uf_n_fft_feat,
                        att_feat=args.uf_n_att_feat,
                        dropout_rate=args.dropout_rate,
                        reduce_method=getattr(args, "uf_reduce_method", "mask"),
                        use_residual=getattr(args, "uf_use_residual", True),
                        use_sub_sampling=getattr(args, "uf_use_subsampling", False),
                        downconv_type=getattr(args, "uf_downconv_type", "conv1d"),
                        use_time_high_dim_as_v=getattr(args, "uf_use_time_high_dim_as_v", False),
                        use_pos_embed=getattr(args, "uf_use_pos_embed", True),
                        pos_embed_type=getattr(args, "uf_pos_embed_type", "2D"),
                        beamformer_type=getattr(args, "uf_beamformer_type", "shengdai"),
                        beamformer_return_mask=getattr(args, "uf_beamformer_return_mask", False),
                        use_abs_pos_pred=getattr(args, "uf_use_abs_pos_pred", True),
                        num_abs_pos_extracted=getattr(args, "uf_num_abs_pos_extracted", 1),
                        attention_method=getattr(args, "uf_attention_method", "channel_wise_att"),
                        power_att_window_len=getattr(args, "uf_power_att_window_len", 2),
                        power_att_affine=getattr(args, "uf_power_att_affine", False),
                        power_att_softmax_factor=getattr(args, "uf_power_att_softmax_factor", 1.0),
                        power_att_auto_grad_softmax_factor=getattr(args, "uf_power_att_auto_grad_softmax_factor", False),
                        power_att_use_delta_power_att=getattr(args, "uf_power_att_use_delta_power_att", False), 
                        power_att_auto_grad_delta_factor=getattr(args, "uf_power_att_auto_grad_delta_factor", True),
                        power_att_delta_factor=getattr(args, "uf_power_att_delta_factor", 3.0),
                        power_att_delta_pos=getattr(args, "uf_power_att_delta_pos", 4),
                        abs_pos_loss_factor=getattr(args, "uf_abs_pos_loss_factor", 50.),
                        use_pos_embed_in_beamform_layer=getattr(args, "uf_use_pos_embed_in_beamform_layer", False),
                        conv_layer_dilation=eval(getattr(args, "cb_conv_layer_dilation", "[1]*8")),
                        no_residual=getattr(args, "cb_conv_layer_no_residual_connect", False),
                        no_residual_no_padding=getattr(args, "cb_conv_layer_no_residual_no_padding", False),
                    )
                elif getattr(args, "use_cnn_front_attention_mean_dilation_res", False):
                    from espnet.nets.pytorch_backend.frontends.cnn_front_attention_mean_dilation_res import CNN_Front_Attention_Mean
                    self.beamformer = CNN_Front_Attention_Mean(
                        conv_layer_list=cb_conv_layer_list,
                        inplane=cb_inplane,
                        use_dilation=args.cb_dilation_replace_stride,
                        n_att_head=args.uf_n_att_head,
                        n_att_blocks=args.uf_n_att_blocks,
                        n_channel_att_blocks=args.uf_n_channel_att_blocks,
                        n_time_att_blocks=args.uf_n_time_att_blocks,
                        fft_feat=args.uf_n_fft_feat,
                        att_feat=args.uf_n_att_feat,
                        dropout_rate=args.dropout_rate,
                        reduce_method=getattr(args, "uf_reduce_method", "mask"),
                        use_residual=getattr(args, "uf_use_residual", True),
                        use_sub_sampling=getattr(args, "uf_use_subsampling", False),
                        downconv_type=getattr(args, "uf_downconv_type", "conv1d"),
                        use_time_high_dim_as_v=getattr(args, "uf_use_time_high_dim_as_v", False),
                        use_pos_embed=getattr(args, "uf_use_pos_embed", True),
                        pos_embed_type=getattr(args, "uf_pos_embed_type", "2D"),
                        beamformer_type=getattr(args, "uf_beamformer_type", "shengdai"),
                        beamformer_return_mask=getattr(args, "uf_beamformer_return_mask", False),
                        use_abs_pos_pred=getattr(args, "uf_use_abs_pos_pred", True),
                        num_abs_pos_extracted=getattr(args, "uf_num_abs_pos_extracted", 1),
                        attention_method=getattr(args, "uf_attention_method", "channel_wise_att"),
                        power_att_window_len=getattr(args, "uf_power_att_window_len", 2),
                        power_att_affine=getattr(args, "uf_power_att_affine", False),
                        power_att_softmax_factor=getattr(args, "uf_power_att_softmax_factor", 1.0),
                        power_att_auto_grad_softmax_factor=getattr(args, "uf_power_att_auto_grad_softmax_factor", False),
                        power_att_use_delta_power_att=getattr(args, "uf_power_att_use_delta_power_att", False), 
                        power_att_auto_grad_delta_factor=getattr(args, "uf_power_att_auto_grad_delta_factor", True),
                        power_att_delta_factor=getattr(args, "uf_power_att_delta_factor", 3.0),
                        power_att_delta_pos=getattr(args, "uf_power_att_delta_pos", 4),
                        abs_pos_loss_factor=getattr(args, "uf_abs_pos_loss_factor", 50.),
                        use_pos_embed_in_beamform_layer=getattr(args, "uf_use_pos_embed_in_beamform_layer", False),
                        conv_layer_dilation=eval(getattr(args, "cb_conv_layer_dilation", "[1]*8")),
                        no_residual=getattr(args, "cb_conv_layer_no_residual_connect", False),
                        add_orig_spectrum_to_mid=getattr(args, "cb_add_orig_spectrum_to_mid", -1),
                        use_freq_embedding=getattr(args, "cb_use_freq_embedding", False),
                    )
                else:
                    from espnet.nets.pytorch_backend.frontends.universal_frontend import Universal_Frontend
                    self.beamformer = Universal_Frontend(
                        conv_layer_list=cb_conv_layer_list,
                        inplane=cb_inplane,
                        use_dilation=args.cb_dilation_replace_stride,
                        n_att_head=args.uf_n_att_head,
                        n_att_blocks=args.uf_n_att_blocks,
                        fft_feat=args.uf_n_fft_feat,
                        att_feat=args.uf_n_att_feat,
                        dropout_rate=args.dropout_rate,
                        reduce_method=getattr(args, "uf_reduce_method", "mask"),
                    )
            else:
                from espnet.nets.pytorch_backend.frontends.dnn_beamformer import DNN_Beamformer
                self.beamformer = DNN_Beamformer(
                    btype=btype,
                    bidim=idim,
                    bunits=bunits,
                    bprojs=bprojs,
                    blayers=blayers,
                    bnmask=bnmask,
                    dropout_rate=bdropout_rate,
                    badim=badim,
                    ref_channel=ref_channel,
                    mono_process=(getattr(args,"mono_process",False) or self.random_pick_channel<0 or self.random_pick_channel==1),
                    mono_process_type=getattr(args, "mono_process_type", 1),
                )
            
        else:
            self.beamformer = None

    def forward(
        self, x: ComplexTensor, ilens: Union[torch.LongTensor, numpy.ndarray, List[int]],
        x_no_reverb=None, x_clean=None,
    ) -> Tuple[ComplexTensor, torch.LongTensor, Optional[ComplexTensor]]:
        assert len(x) == len(ilens), (len(x), len(ilens))
        # (B, T, F) or (B, T, C, F)
        if x.dim() not in (3, 4):
            raise ValueError(f"Input dim must be 3 or 4: {x.dim()}")
        if not torch.is_tensor(ilens):
            ilens = torch.from_numpy(numpy.asarray(ilens)).to(x.device)

        mask = None
        h = x
        use_beamformer = False
        if h.dim() == 4: # or (self.use_complex_beamformer and h.dim()==3):
            if self.noise_to_backend and self.training:
                choices = [(False, False)] if not self.use_frontend_for_all else []
                if self.use_wpe:
                    choices.append((True, False))

                if self.use_beamformer:
                    choices.append((False, True))

                use_wpe, use_beamformer = choices[numpy.random.randint(len(choices))]

            else:
                use_wpe = self.use_wpe
                use_beamformer = self.use_beamformer

            # randomly pick given number of channels of audio into frontend
            if self.random_pick_channel != 0 and self.training:
                if self.use_complex_beamformer or self.use_universal_beamformer:
                    C = h.shape[1] # B C T F
                    if C==1:
                        pass
                    elif self.random_pick_channel == 2:
                        h = h.index_select(1, torch.randperm(C).cuda()[0:2])   # We use 1,3 microphone for 2 mic track; Actually the 2nd channel here is correspond to the 3th mic.
                    else:
                        if self.random_pick_channel > 0:
                            num_channels = self.random_pick_channel
                        elif self.random_pick_channel == -1:
                            num_channels = int(torch.randint(2, C+1, (1,)))
                        elif self.random_pick_channel == -2:
                            num_channels = int(torch.randint(1, C+1, (1,)))
                        elif self.random_pick_channel == -3:
                            num_channels = C if numpy.random.randint(2) == 0 else 1 
                        else:
                            raise ValueError(self.random_pick_channel)
                        # num_channels = int(self.random_pick_channel if self.random_pick_channel > 0 else torch.randint(2, C+1, (1,)))
                        channel_setoff = torch.randint(0, C-num_channels+1, (1,))
                        h = h[:, channel_setoff: channel_setoff+num_channels]

                else:
                    C = h.shape[2] # B T C F
                    if C==1:
                        pass
                    elif self.random_pick_channel == 2:
                        h = h.index_select(2, torch.randperm(C).cuda()[0:2])  # We use 1,3 microphone for 2 mic track
                    else:
                        # num_channels = int(self.random_pick_channel if self.random_pick_channel > 0 else torch.randint(2, C+1, (1,)))
                        if self.random_pick_channel > 0:
                            num_channels = self.random_pick_channel
                        elif self.random_pick_channel == -1:
                            num_channels = int(torch.randint(2, C+1, (1,)))
                        elif self.random_pick_channel == -2:
                            num_channels = int(torch.randint(1, C+1, (1,)))
                        elif self.random_pick_channel == -3:
                            num_channels = C if numpy.random.randint(2) == 0 else 1 
                        else:
                            raise ValueError(self.random_pick_channel)
                        # if num_channels == 1: # Baseline doesn't surpport mono-channel input, so we just skip it.
                        #     return h, ilens, None
                        channel_setoff = torch.randint(0, C-num_channels+1, (1,))
                        h = h[:, :, channel_setoff: channel_setoff+num_channels]
                    

            # 1. WPE
            if use_wpe:
                # h: (B, T, C, F) -> h: (B, T, C, F)
                h, ilens, mask = self.wpe(h, ilens)

            # 2. Beamformer
            if use_beamformer:
                # h: (B, T, C, F) -> h: (B, T, F)
                if not self.use_complex_beamformer:
                    h, ilens, mask = self.beamformer(h, ilens)
                else:
                    h, ilens, loss, loss_reverb, loss_clean = self.beamformer(h, ilens, x_no_reverb, x_clean)
            # else:
            #     if self.use_complex_beamformer:
            #         h = h.contiguous().view(h.shape[0],-1,4,h.shape[-1])

        if not self.use_complex_beamformer:   
            return h, ilens, mask
        else:
            if use_beamformer:
                return h, ilens, loss, loss_reverb, loss_clean
            else:
                return h, ilens, None, None, None


def frontend_for(args, idim):
    if args.cb_use_clean_loss:
        print("Note: We use MSELoss with clean wav's STFT to make frontend converge.")
    return Frontend(
        args=args,
        idim=idim,
        # WPE options
        use_wpe=args.use_wpe,
        wtype=args.wtype,
        wlayers=args.wlayers,
        wunits=args.wunits,
        wprojs=args.wprojs,
        wdropout_rate=args.wdropout_rate,
        taps=args.wpe_taps,
        delay=args.wpe_delay,
        use_dnn_mask_for_wpe=args.use_dnn_mask_for_wpe,
        # Beamformer options
        use_beamformer=args.use_beamformer,
        use_beamformer_guo=args.use_beamformer_guo,
        use_complex_beamformer=args.use_complex_beamformer,
        no_reverb = getattr(args, "no_reverb", False),
        btype=args.btype,
        blayers=args.blayers,
        bunits=args.bunits,
        bprojs=args.bprojs,
        bnmask=args.bnmask,
        badim=args.badim,
        ref_channel=args.ref_channel,
        bdropout_rate=args.bdropout_rate,
        cb_conv_layer_list=eval(args.cb_conv_layer_list),
        cb_inplane=args.cb_inplane, 
        cb_num_dereverb_frames=args.cb_num_dereverb_frames, 
        cb_num_dereverb_blocks=args.cb_num_dereverb_blocks, 
        cb_use_dereverb_loss=args.cb_use_dereverb_loss, 
        cb_use_clean_loss=args.cb_use_clean_loss, 
        cb_ratio_reverb=args.cb_ratio_reverb,
        cb_down_sample=getattr(args, "cb_down_sample", False),
        cb_down_sample_layer=eval(getattr(args, "cb_down_sample_layer", "[(16, 7, 2), (64, 5, 2)]")),
    )
