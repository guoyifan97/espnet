# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer speech recognition model (pytorch)."""

from argparse import Namespace
from distutils.util import strtobool
# from espnet.transform.spec_augment import spec_augment
from espnet2.asr.specaug.specaug import SpecAug
from itertools import groupby
import logging
import math

import numpy
import torch

from espnet.nets.asr_interface import ASRInterface
from espnet.nets.ctc_prefix_score import CTCPrefixScore
from espnet.nets.e2e_asr_common import end_detect
from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.e2e_asr import CTC_LOSS_THRESHOLD
# from espnet.nets.pytorch_backend.e2e_asr import Reporter
from espnet.nets.pytorch_backend.nets_utils import get_subsample
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.rnn.decoders import CTC_SCORING_RATIO
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
# from espnet.nets.pytorch_backend.transformer.add_sos_eos import mask_uniform
from espnet.nets.pytorch_backend.transformer.argument import (
    add_arguments_transformer_common,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.attention import (
    MultiHeadedAttention,  # noqa: H301
    RelPositionMultiHeadedAttention,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from espnet.nets.pytorch_backend.transformer.dynamic_conv import DynamicConvolution
from espnet.nets.pytorch_backend.transformer.dynamic_conv2d import DynamicConvolution2D
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.initializer import initialize
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet.nets.pytorch_backend.transformer.plot import PlotAttentionReport
from espnet.nets.scorers.ctc import CTCPrefixScorer
from espnet.utils.fill_missing_args import fill_missing_args

import argparse
from itertools import groupby
import logging
import math
import os

import chainer
from chainer import reporter
import editdistance
import numpy as np
import six
import torch

from espnet.nets.asr_interface import ASRInterface
from espnet.nets.e2e_asr_common import label_smoothing_dist
from espnet.nets.pytorch_backend.ctc import ctc_for
from espnet.nets.pytorch_backend.frontends.feature_transform import (
    feature_transform_for,  # noqa: H301
)
from espnet.nets.pytorch_backend.frontends.feature_transform_complex import (
    feature_transform_for_complex,  # noqa: H301
)
from espnet.nets.pytorch_backend.frontends.frontend import frontend_for
from espnet.nets.pytorch_backend.initialization import lecun_normal_init_parameters
from espnet.nets.pytorch_backend.initialization import set_forget_bias_to_one
from espnet.nets.pytorch_backend.nets_utils import get_subsample
from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet.nets.pytorch_backend.nets_utils import to_torch_tensor
from espnet.nets.pytorch_backend.rnn.attentions import att_for
from espnet.nets.pytorch_backend.rnn.decoders import decoder_for
from espnet.nets.pytorch_backend.rnn.encoders import encoder_for
from espnet.nets.scorers.ctc import CTCPrefixScorer
from espnet.utils.fill_missing_args import fill_missing_args

class Reporter(chainer.Chain):
    """A chainer reporter wrapper."""

    def report(self, loss_ctc, loss_att, acc, cer_ctc, cer, wer, mtl_loss, loss_reverb=None, loss_clean=None, frontend_loss=None):
        """Report at every step."""
        reporter.report({"loss_ctc": loss_ctc}, self)
        reporter.report({"loss_att": loss_att}, self)
        reporter.report({"acc": acc}, self)
        reporter.report({"cer_ctc": cer_ctc}, self)
        reporter.report({"cer": cer}, self)
        reporter.report({"wer": wer}, self)
        logging.info("mtl loss:" + str(mtl_loss))
        reporter.report({"loss": mtl_loss}, self)
        if isinstance(loss_reverb, float):
            reporter.report({"loss_reverb": loss_reverb}, self)
        if isinstance(loss_clean, float):
            reporter.report({"loss_clean": loss_clean}, self)
        if isinstance(frontend_loss, float):
            reporter.report({"frontend_loss": frontend_loss}, self)

class FrontendLo(torch.nn.Module):
    def __init__(self, idim, odim, downsample=False):
        super().__init__()
        self.downsample = downsample
        if downsample:
            self.pool = torch.nn.AvgPool1d(4, 4)
        self.linear = torch.nn.Linear(idim, odim)
        self.norm = torch.nn.InstanceNorm2d(1)
        self.relu = torch.nn.ReLU()
    

    def forward(self, x, ilen):
        if self.downsample:
            x = self.pool(x.transpose(1,2).contiguous()).transpose(1,2).contiguous()
            ilen = ilen//4
        x = self.linear(x)
        x = self.norm(x.unsqueeze(1)).squeeze(1)
        x = self.relu(x)
        
        return x, ilen

class E2E(ASRInterface, torch.nn.Module):
    """E2E module.

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options

    """

    @staticmethod
    def add_arguments(parser):
        """Add arguments."""
        group = parser.add_argument_group("transformer model setting")

        group = add_arguments_transformer_common(group)
        # group.add_argument(
        #     "--transformer-init",
        #     type=str,
        #     default="pytorch",
        #     choices=[
        #         "pytorch",
        #         "xavier_uniform",
        #         "xavier_normal",
        #         "kaiming_uniform",
        #         "kaiming_normal",
        #     ],
        #     help="how to initialize transformer parameters",
        # )
        # group.add_argument(
        #     "--transformer-input-layer",
        #     type=str,
        #     default="conv2d",
        #     choices=["conv2d", "linear", "embed", "vgg2l"],
        #     help="transformer input layer type",
        # )
        # group.add_argument(
        #     "--transformer-attn-dropout-rate",
        #     default=None,
        #     type=float,
        #     help="dropout in transformer attention. use --dropout-rate if None is set",
        # )
        # group.add_argument(
        #     "--transformer-lr",
        #     default=10.0,
        #     type=float,
        #     help="Initial value of learning rate",
        # )
        # group.add_argument(
        #     "--transformer-warmup-steps",
        #     default=25000,
        #     type=int,
        #     help="optimizer warmup steps",
        # )
        # group.add_argument(
        #     "--transformer-length-normalized-loss",
        #     default=True,
        #     type=strtobool,
        #     help="normalize loss by length",
        # )
        # group.add_argument(
        #     "--transformer-encoder-selfattn-layer-type",
        #     type=str,
        #     default="selfattn",
        #     choices=[
        #         "selfattn",
        #         "rel_selfattn",
        #         "lightconv",
        #         "lightconv2d",
        #         "dynamicconv",
        #         "dynamicconv2d",
        #         "light-dynamicconv2d",
        #     ],
        #     help="transformer encoder self-attention layer type",
        # )
        # group.add_argument(
        #     "--transformer-decoder-selfattn-layer-type",
        #     type=str,
        #     default="selfattn",
        #     choices=[
        #         "selfattn",
        #         "lightconv",
        #         "lightconv2d",
        #         "dynamicconv",
        #         "dynamicconv2d",
        #         "light-dynamicconv2d",
        #     ],
        #     help="transformer decoder self-attention layer type",
        # )
        # # Lightweight/Dynamic convolution related parameters.
        # # See https://arxiv.org/abs/1912.11793v2
        # # and https://arxiv.org/abs/1901.10430 for detail of the method.
        # # Configurations used in the first paper are in
        # # egs/{csj, librispeech}/asr1/conf/tuning/ld_conv/
        # group.add_argument(
        #     "--wshare",
        #     default=4,
        #     type=int,
        #     help="Number of parameter shargin for lightweight convolution",
        # )
        # group.add_argument(
        #     "--ldconv-encoder-kernel-length",
        #     default="21_23_25_27_29_31_33_35_37_39_41_43",
        #     type=str,
        #     help="kernel size for lightweight/dynamic convolution: "
        #     'Encoder side. For example, "21_23_25" means kernel length 21 for '
        #     "First layer, 23 for Second layer and so on.",
        # )
        # group.add_argument(
        #     "--ldconv-decoder-kernel-length",
        #     default="11_13_15_17_19_21",
        #     type=str,
        #     help="kernel size for lightweight/dynamic convolution: "
        #     'Decoder side. For example, "21_23_25" means kernel length 21 for '
        #     "First layer, 23 for Second layer and so on.",
        # )
        # group.add_argument(
        #     "--ldconv-usebias",
        #     type=strtobool,
        #     default=False,
        #     help="use bias term in lightweight/dynamic convolution",
        # )
        # group.add_argument(
        #     "--dropout-rate",
        #     default=0.0,
        #     type=float,
        #     help="Dropout rate for the encoder",
        # )
        # # Encoder
        # group.add_argument(
        #     "--elayers",
        #     default=4,
        #     type=int,
        #     help="Number of encoder layers (for shared recognition part "
        #     "in multi-speaker asr mode)",
        # )
        # group.add_argument(
        #     "--eunits",
        #     "-u",
        #     default=300,
        #     type=int,
        #     help="Number of encoder hidden units",
        # )
        # # Attention
        # group.add_argument(
        #     "--adim",
        #     default=320,
        #     type=int,
        #     help="Number of attention transformation dimensions",
        # )
        # group.add_argument(
        #     "--aheads",
        #     default=4,
        #     type=int,
        #     help="Number of heads for multi head attention",
        # )
        # # Decoder
        # group.add_argument(
        #     "--dlayers", default=1, type=int, help="Number of decoder layers"
        # )
        # group.add_argument(
        #     "--dunits", default=320, type=int, help="Number of decoder hidden units"
        # )
        # # Non-autoregressive training
        # group.add_argument(
        #     "--decoder-mode",
        #     default="AR",
        #     type=str,
        #     choices=["ar", "maskctc"],
        #     help="AR: standard autoregressive training, "
        #     "maskctc: non-autoregressive training based on Mask CTC",
        # )
        return parser

    @property
    def attention_plot_class(self):
        """Return PlotAttentionReport."""
        return PlotAttentionReport
    
    def get_total_subsampling_factor(self):
        """Get total subsampling factor."""
        return self.encoder.conv_subsampling_factor * int(numpy.prod(self.subsample))


    def __init__(self, idim, odim, args, ignore_id=-1):
        """Construct an E2E object.

        :param int idim: dimension of inputs
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        torch.nn.Module.__init__(self)

        # fill missing arguments for compatibility
        args = fill_missing_args(args, self.add_arguments)
        
        self.use_complex_beamformer = getattr(args, "use_complex_beamformer", False)
        self.use_universal_beamformer = getattr(args, "use_universal_beamformer", False)
        # ADDED GUO
        if getattr(args, "use_frontend", False):  # use getattr to keep compatibility
            self.frontend = frontend_for(args, idim)
            # if getattr(args, "use_rand_fttransform", False): #  self.use_complex_beamformer or self.use_universal_beamformer: #use-complex-beamformernew
            
            if getattr(args, "not_use_mel_transform", True) and (self.use_complex_beamformer or self.use_universal_beamformer): #use-complex-beamformernew
            # if self.use_complex_beamformer:
                self.feature_transform = feature_transform_for_complex(args, getattr(args, "no_padding_idim", (idim-1) * 2), getattr(args, "feature_reduce", False))
                
                if args.cb_down_sample:
                    self.cb_down_sample = args.cb_down_sample
                    
                    for _, kernel_size, stride in eval(args.cb_down_sample_layer):
                        kernel_size = kernel_size if isinstance(kernel_size, int) else kernel_size[1]
                        idim = np.ceil((idim-kernel_size+1) / stride)
                    idim = int(idim)
                if getattr(args, "feature_reduce", False):
                    idim = args.n_mels
                
            else:
                self.feature_transform = feature_transform_for(args, (idim - 1) * 2)
                idim = args.n_mels
            
            # ###
            # self.feature_transform = feature_transform_for(args, (idim - 1) * 2)
            # idim = args.n_mels
            # ###

            self.idim = idim

        else:
            self.frontend = None
        
        if getattr(args, "cb_use_specaug_after_frontend", False):
            self.use_specaug = True
            self.specaug_module = SpecAug()


        if getattr(args, "cb_use_frontend_ctc", False):
            logging.warning("We use frontend ctc to make the network converge.")
            if getattr(args, "cb_use_frontend_ctc_downsample", False):
                self.frontend_lo = FrontendLo(idim, idim, True)
            # else:
            #     self.frontend_lo = FrontendLo(idim, idim, False)
            logging.warning(f"We use the PHONE level frontend ctc loss to make frontend coverge.({getattr(args,'cb_frontend_ctc_odim',5002)-2} phoneme including <unk> plus <eos> and <blank>, {getattr(args,'cb_frontend_ctc_odim',5002)} in total)")
            self.frontend_ctc = CTC(
                getattr(args,"cb_frontend_ctc_odim", 5002), idim, args.dropout_rate, ctc_type=args.ctc_type, reduce=True
            )

        if args.transformer_attn_dropout_rate is None:
            args.transformer_attn_dropout_rate = args.dropout_rate
        self.intermediate_ctc_weight = args.intermediate_ctc_weight
        self.intermediate_ctc_layers = []
        if args.intermediate_ctc_layer != "":
            self.intermediate_ctc_layers = [
                int(i) for i in args.intermediate_ctc_layer.split(",")
            ]

        self.encoder = Encoder(
            idim=idim,
            selfattention_layer_type=args.transformer_encoder_selfattn_layer_type,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            conv_wshare=args.wshare,
            conv_kernel_length=args.ldconv_encoder_kernel_length,
            conv_usebias=args.ldconv_usebias,
            linear_units=args.eunits,
            num_blocks=args.elayers,
            input_layer=args.transformer_input_layer,
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            attention_dropout_rate=args.transformer_attn_dropout_rate,
            stochastic_depth_rate=args.stochastic_depth_rate,
            intermediate_layers=self.intermediate_ctc_layers,
        )
        if args.mtlalpha < 1:
            self.decoder = Decoder(
                odim=odim,
                selfattention_layer_type=args.transformer_decoder_selfattn_layer_type,
                attention_dim=args.adim,
                attention_heads=args.aheads,
                conv_wshare=args.wshare,
                conv_kernel_length=args.ldconv_decoder_kernel_length,
                conv_usebias=args.ldconv_usebias,
                linear_units=args.dunits,
                num_blocks=args.dlayers,
                dropout_rate=args.dropout_rate,
                positional_dropout_rate=args.dropout_rate,
                self_attention_dropout_rate=args.transformer_attn_dropout_rate,
                src_attention_dropout_rate=args.transformer_attn_dropout_rate,
            )
            self.criterion = LabelSmoothingLoss(
                odim,
                ignore_id,
                args.lsm_weight,
                args.transformer_length_normalized_loss,
            )
        else:
            self.decoder = None
            self.criterion = None
        self.blank = 0
        # self.decoder_mode = args.decoder_mode
        # if self.decoder_mode == "maskctc":
        #     self.mask_token = odim - 1
        #     self.sos = odim - 2
        #     self.eos = odim - 2
        # else:
        self.sos = odim - 1
        self.eos = odim - 1
        self.odim = odim
        self.ignore_id = ignore_id
        self.subsample = get_subsample(args, mode="asr", arch="transformer")
        self.reporter = Reporter()

        self.reset_parameters(args)
        self.adim = args.adim  # used for CTC (equal to d_model)
        self.mtlalpha = args.mtlalpha
        if args.mtlalpha > 0.0:
            self.ctc = CTC(
                odim, args.adim, args.dropout_rate, ctc_type=args.ctc_type, reduce=True
            )
        else:
            self.ctc = None

        
        
        if args.report_cer or args.report_wer:
            self.error_calculator = ErrorCalculator(
                args.char_list,
                args.sym_space,
                args.sym_blank,
                args.report_cer,
                args.report_wer,
            )
        else:
            self.error_calculator = None
        self.rnnlm = None

        if self.use_complex_beamformer or self.use_universal_beamformer:
            # assert args.complex_loss_ratio<=1 and args.complex_loss_ratio>=0, "%f"%args.complex_loss_ratio
            self.complex_loss_ratio = args.complex_loss_ratio
            self.ratio_reverb = args.cb_ratio_reverb
        

    def reset_parameters(self, args):
        """Initialize parameters."""
        # initialize parameters
        initialize(self, args.transformer_init)

    def forward(self, xs_pad, ilens, ys_pad):
        """E2E forward.

        :param torch.Tensor xs_pad: batch of padded source sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of source sequences (B)
        :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
        :return: ctc loss value
        :rtype: torch.Tensor
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy in attention decoder
        :rtype: float
        """
        # GUO
        # 0. Frontend input is stft
        if getattr(self, "frontend_ctc", False):
            ys_pad_phone = ys_pad[1]
            ys_pad = ys_pad[0]
        
        if self.frontend is not None:
            if not self.use_complex_beamformer and not self.use_universal_beamformer:
                xs_pad = to_torch_tensor(xs_pad)
                hs_pad, hlens, mask = self.frontend(xs_pad[0] if isinstance(xs_pad, list) else xs_pad, ilens)
                frontend_loss, loss_reverb, loss_clean = None, None, None
            elif self.use_universal_beamformer:
                xs_pad = to_torch_tensor(xs_pad)
                hs_pad, hlens, frontend_loss = self.frontend(xs_pad[0] if isinstance(xs_pad, list) else xs_pad, ilens)
                loss_reverb, loss_clean = None, None
            else:
                xs_pad = to_torch_tensor(xs_pad)
                hs_pad, hlens, frontend_loss, loss_reverb, loss_clean = self.frontend(xs_pad, ilens)
                # if len(xs_pad)==1:  # 这些是当时输入既有reverb，noreverb，clean时的，现在data.json都没这些了
                #     hs_pad, hlens, frontend_loss, loss_reverb, loss_clean = self.frontend(xs_pad[0], ilens)

                # else:
                #     hs_pad, hlens, frontend_loss, loss_reverb, loss_clean = self.frontend(xs_pad[0], ilens, xs_pad[1], xs_pad[2])

            del xs_pad, ilens

            hs_pad, hlens = self.feature_transform(hs_pad, hlens)

            if getattr(self, "use_specaug", False) and self.training:
                hs_pad = self.specaug_module(hs_pad)[0]
            
        else:
            hs_pad, hlens = xs_pad, ilens
        
        
        
        # if self.training:
        #     return frontend_loss

        # 1. forward encoder
        
        hs_pad = hs_pad[:, : max(hlens)]  # for data parallel
        src_mask = make_non_pad_mask(hlens.tolist()).to(hs_pad.device).unsqueeze(-2)
        
        # if getattr(self, "frontend_ctc", False):
        #     if getattr(self, "frontend_lo", False):
        #         out_hs, out_hlen = self.frontend_lo(hs_pad, hlens)
        #         frontend_loss = self.frontend_ctc(out_hs.view(out_hs.size(0), -1, self.idim), out_hlen, ys_pad_phone)
        #     else:    
        #         frontend_loss = self.frontend_ctc(hs_pad.view(hs_pad.size(0), -1, self.idim), hlens, ys_pad_phone)

        hs_pad, hs_mask, hs_intermediates = self.encoder(hs_pad, src_mask)

        del src_mask

        self.hs_pad = hs_pad
        

        # 2. forward decoder
        if self.decoder is not None:
            # if self.decoder_mode == "maskctc":
            #     ys_in_pad, ys_out_pad = mask_uniform(
            #         ys_pad, self.mask_token, self.eos, self.ignore_id
            #     )
            #     ys_mask = (ys_in_pad != self.ignore_id).unsqueeze(-2)
                
            # else:
            ys_in_pad, ys_out_pad = add_sos_eos(
                ys_pad, self.sos, self.eos, self.ignore_id
            )
            ys_mask = target_mask(ys_in_pad, self.ignore_id)
                

            pred_pad, pred_mask = self.decoder(ys_in_pad, ys_mask, hs_pad, hs_mask)
            self.pred_pad = pred_pad

            del ys_in_pad, ys_mask, pred_mask

            # 3. compute attention loss
            loss_att = self.criterion(pred_pad, ys_out_pad)
            self.acc = th_accuracy(
                pred_pad.view(-1, self.odim), ys_out_pad, ignore_label=self.ignore_id
            )

            del pred_pad, ys_out_pad

        else:
            loss_att = None
            self.acc = None

        # TODO(karita) show predicted text
        # TODO(karita) calculate these stats
        cer_ctc = None
        loss_intermediate_ctc = 0.0
        if self.mtlalpha == 0.0:
            loss_ctc = None
        else:
            # GUO replace
            batch_size = hs_pad.size(0)
            
            hs_len = hs_mask.view(batch_size, -1).sum(1)
            
            del hs_mask

            loss_ctc = self.ctc(hs_pad.view(batch_size, -1, self.adim), hs_len, ys_pad)
            
            if not self.training and self.error_calculator is not None:
                ys_hat = self.ctc.argmax(hs_pad.view(batch_size, -1, self.adim)).data
                cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
            # for visualization
            if not self.training:
                self.ctc.softmax(hs_pad)
            if self.intermediate_ctc_weight > 0 and self.intermediate_ctc_layers:
                for hs_intermediate in hs_intermediates:
                    # assuming hs_intermediates and hs_pad has same length / padding
                    loss_inter = self.ctc(
                        hs_intermediate.view(batch_size, -1, self.adim), hs_len, ys_pad
                    )
                    loss_intermediate_ctc += loss_inter

                loss_intermediate_ctc /= len(self.intermediate_ctc_layers)

        del hs_pad

        # 5. compute cer/wer
        if self.training or self.error_calculator is None or self.decoder is None:
            cer, wer = None, None
        else:
            ys_hat = pred_pad.argmax(dim=-1)
            cer, wer = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        del ys_pad

        # copied from e2e_asr
        alpha = self.mtlalpha
        if alpha == 0:
            self.loss = loss_att
            loss_att_data = float(loss_att)
            loss_ctc_data = None
        elif alpha == 1:
            self.loss = loss_ctc
            if self.intermediate_ctc_weight > 0:
                self.loss = (
                    1 - self.intermediate_ctc_weight
                ) * loss_ctc + self.intermediate_ctc_weight * loss_intermediate_ctc
            loss_att_data = None
            loss_ctc_data = float(loss_ctc)
        else:
            self.loss = alpha * loss_ctc + (1 - alpha) * loss_att
            if self.intermediate_ctc_weight > 0:
                self.loss = (
                    (1 - alpha - self.intermediate_ctc_weight) * loss_att
                    + alpha * loss_ctc
                    + self.intermediate_ctc_weight * loss_intermediate_ctc
                )
            loss_att_data = float(loss_att)
            loss_ctc_data = float(loss_ctc)
            del loss_att, loss_ctc

        loss_data = float(self.loss)
        if loss_data < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
            if self.use_complex_beamformer or self.use_universal_beamformer:
                if isinstance(frontend_loss, torch.Tensor):
                    self.reporter.report(
                        loss_ctc_data, loss_att_data, self.acc, cer_ctc, cer, wer, loss_data, frontend_loss=float(frontend_loss)
                    )
                elif isinstance(loss_reverb, torch.Tensor):
                    self.reporter.report(
                        loss_ctc_data, loss_att_data, self.acc, cer_ctc, cer, wer, loss_data, float(loss_reverb), float(loss_clean)
                    )
                elif isinstance(loss_clean, torch.Tensor):
                    self.reporter.report(
                        loss_ctc_data, loss_att_data, self.acc, cer_ctc, cer, wer, loss_data, float(loss_clean)
                    )
                else:
                    self.reporter.report(
                        loss_ctc_data, loss_att_data, self.acc, cer_ctc, cer, wer, loss_data,
                    )
            else:
                self.reporter.report(
                    loss_ctc_data, loss_att_data, self.acc, cer_ctc, cer, wer, loss_data
                )
        else:
            logging.warning("loss (=%f) is not correct", loss_data)
        
        if self.use_complex_beamformer or self.use_universal_beamformer:
            if isinstance(frontend_loss, torch.Tensor):
                self.loss = (1-self.complex_loss_ratio)*self.loss + self.complex_loss_ratio*frontend_loss
            elif isinstance(loss_clean, torch.Tensor):
                self.loss = self.loss + self.complex_loss_ratio * loss_clean

        
        return self.loss

    def scorers(self):
        """Scorers."""
        return dict(decoder=self.decoder, ctc=CTCPrefixScorer(self.ctc, self.eos))

    def encode(self, x):
        """Encode acoustic features.

        :param ndarray x: source acoustic feature (T, D) (T,C,D)
        :return: encoder outputs
        :rtype: torch.Tensor
        """
        self.eval()

        x = to_torch_tensor(x).unsqueeze(0)
        
        # if not (self.use_complex_beamformer or self.use_universal_beamformer):
        #     ilens = [x.shape[1]]
        # else:
        #     # 1 TxC F -> 1 T C F -> 1 C T F
        #     ilens = [x.shape[1]]
        #     if len(x.shape) == 4:
        #         x = x.transpose(1,2).contiguous()
        ilens = [x.shape[1]]

        # GUO added
        if self.frontend is not None:
            if not self.use_complex_beamformer:
                enhanced, hlens, mask = self.frontend(x, ilens)
            else:
                enhanced, hlens, _, _, _ = self.frontend(x, ilens)

            # enhanced, hlens, mask = self.frontend(x, ilens)
            hs, hlens = self.feature_transform(enhanced, hlens)
        else:
            hs, hlens = x, ilens

        # GUO
        enc_output, _, _ = self.encoder(hs, None)
        # enc_output, _ = self.encoder(x, None)


        return enc_output.squeeze(0)

    def recognize(self, x, recog_args, char_list=None, rnnlm=None, use_jit=False):
        """Recognize input speech.

        :param ndnarray x: input acoustic feature (B, T, D) or (T, D)
        :param Namespace recog_args: argment Namespace contraining options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        # T C F -> T C' F
        if getattr(recog_args, "random_pick_channel", False):
            if recog_args.random_pick_channel != 0:
                logging.warning(f"We shall pick {recog_args.random_pick_channel} channel{'s' if recog_args.random_pick_channel!=1 else ''} randomly to decode")
                C = x.shape[1] # T C F
                if C==1:
                    pass
                elif recog_args.random_pick_channel == 2:
                    num_channels = 2
                    x = x[:, [i<2 for i in numpy.random.permutation(C)]]
                else:
                    if recog_args.random_pick_channel > 0:
                        num_channels = recog_args.random_pick_channel
                    elif recog_args.random_pick_channel == -1:
                        num_channels = int(torch.randint(2, C+1, (1,)))
                    elif recog_args.random_pick_channel == -2:
                        num_channels = int(torch.randint(1, C+1, (1,)))
                    elif recog_args.random_pick_channel == -3:
                        num_channels = C if numpy.random.randint(2) == 0 else 1 
                    else:
                        raise ValueError(recog_args.random_pick_channel)
                    channel_setoff = torch.randint(0, C-num_channels+1, (1,))
                    x = x[:, channel_setoff: channel_setoff+num_channels]
                logging.info(f"This audio is decode with {num_channels} channel{'s' if num_channels > 1 else ''}")
            
        enc_output = self.encode(x).unsqueeze(0)
        if self.mtlalpha == 1.0:
            recog_args.ctc_weight = 1.0
            logging.info("Set to pure CTC decoding mode.")

        if self.mtlalpha > 0 and recog_args.ctc_weight == 1.0:
            from itertools import groupby

            lpz = self.ctc.argmax(enc_output)
            collapsed_indices = [x[0] for x in groupby(lpz[0])]
            hyp = [x for x in filter(lambda x: x != self.blank, collapsed_indices)]
            nbest_hyps = [{"score": 0.0, "yseq": [self.sos] + hyp}]
            if recog_args.beam_size > 1:
                raise NotImplementedError("Pure CTC beam search is not implemented.")
            # TODO(hirofumi0810): Implement beam search
            return nbest_hyps
        elif self.mtlalpha > 0 and recog_args.ctc_weight > 0.0:
            lpz = self.ctc.log_softmax(enc_output)
            lpz = lpz.squeeze(0)
        else:
            lpz = None

        h = enc_output.squeeze(0)

        logging.info("input lengths: " + str(h.size(0)))
        # search parms
        beam = recog_args.beam_size
        penalty = recog_args.penalty
        ctc_weight = recog_args.ctc_weight

        # preprare sos
        y = self.sos
        vy = h.new_zeros(1).long()

        if recog_args.maxlenratio == 0:
            maxlen = h.shape[0]
        else:
            # maxlen >= 1
            maxlen = max(1, int(recog_args.maxlenratio * h.size(0)))
        minlen = int(recog_args.minlenratio * h.size(0))
        logging.info("max output length: " + str(maxlen))
        logging.info("min output length: " + str(minlen))

        # initialize hypothesis
        if rnnlm:
            hyp = {"score": 0.0, "yseq": [y], "rnnlm_prev": None}
        else:
            hyp = {"score": 0.0, "yseq": [y]}
        if lpz is not None:
            ctc_prefix_score = CTCPrefixScore(lpz.detach().numpy(), 0, self.eos, numpy)
            hyp["ctc_state_prev"] = ctc_prefix_score.initial_state()
            hyp["ctc_score_prev"] = 0.0
            if ctc_weight != 1.0:
                # pre-pruning based on attention scores
                ctc_beam = min(lpz.shape[-1], int(beam * CTC_SCORING_RATIO))
            else:
                ctc_beam = lpz.shape[-1]
        hyps = [hyp]
        ended_hyps = []

        import six

        traced_decoder = None
        for i in six.moves.range(maxlen):
            logging.debug("position " + str(i))

            hyps_best_kept = []
            for hyp in hyps:
                vy[0] = hyp["yseq"][i]

                # get nbest local scores and their ids
                ys_mask = subsequent_mask(i + 1).unsqueeze(0)
                ys = torch.tensor(hyp["yseq"]).unsqueeze(0)
                # FIXME: jit does not match non-jit result
                if use_jit:
                    if traced_decoder is None:
                        traced_decoder = torch.jit.trace(
                            self.decoder.forward_one_step, (ys, ys_mask, enc_output)
                        )
                    local_att_scores = traced_decoder(ys, ys_mask, enc_output)[0]
                else:
                    local_att_scores = self.decoder.forward_one_step(
                        ys, ys_mask, enc_output
                    )[0]

                if rnnlm:
                    rnnlm_state, local_lm_scores = rnnlm.predict(hyp["rnnlm_prev"], vy)
                    local_scores = (
                        local_att_scores + recog_args.lm_weight * local_lm_scores
                    )
                else:
                    local_scores = local_att_scores

                if lpz is not None:
                    local_best_scores, local_best_ids = torch.topk(
                        local_att_scores, ctc_beam, dim=1
                    )
                    ctc_scores, ctc_states = ctc_prefix_score(
                        hyp["yseq"], local_best_ids[0], hyp["ctc_state_prev"]
                    )
                    local_scores = (1.0 - ctc_weight) * local_att_scores[
                        :, local_best_ids[0]
                    ] + ctc_weight * torch.from_numpy(
                        ctc_scores - hyp["ctc_score_prev"]
                    )
                    if rnnlm:
                        local_scores += (
                            recog_args.lm_weight * local_lm_scores[:, local_best_ids[0]]
                        )
                    local_best_scores, joint_best_ids = torch.topk(
                        local_scores, beam, dim=1
                    )
                    local_best_ids = local_best_ids[:, joint_best_ids[0]]
                else:
                    local_best_scores, local_best_ids = torch.topk(
                        local_scores, beam, dim=1
                    )

                for j in six.moves.range(beam):
                    new_hyp = {}
                    new_hyp["score"] = hyp["score"] + float(local_best_scores[0, j])
                    new_hyp["yseq"] = [0] * (1 + len(hyp["yseq"]))
                    new_hyp["yseq"][: len(hyp["yseq"])] = hyp["yseq"]
                    new_hyp["yseq"][len(hyp["yseq"])] = int(local_best_ids[0, j])
                    if rnnlm:
                        new_hyp["rnnlm_prev"] = rnnlm_state
                    if lpz is not None:
                        new_hyp["ctc_state_prev"] = ctc_states[joint_best_ids[0, j]]
                        new_hyp["ctc_score_prev"] = ctc_scores[joint_best_ids[0, j]]
                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)

                hyps_best_kept = sorted(
                    hyps_best_kept, key=lambda x: x["score"], reverse=True
                )[:beam]

            # sort and get nbest
            hyps = hyps_best_kept
            logging.debug("number of pruned hypothes: " + str(len(hyps)))
            if char_list is not None:
                logging.debug(
                    "best hypo: "
                    + "".join([char_list[int(x)] for x in hyps[0]["yseq"][1:]])
                )

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                logging.info("adding <eos> in the last position in the loop")
                for hyp in hyps:
                    hyp["yseq"].append(self.eos)

            # add ended hypothes to a final list, and removed them from current hypothes
            # (this will be a probmlem, number of hyps < beam)
            remained_hyps = []
            for hyp in hyps:
                if hyp["yseq"][-1] == self.eos:
                    # only store the sequence that has more than minlen outputs
                    # also add penalty
                    if len(hyp["yseq"]) > minlen:
                        hyp["score"] += (i + 1) * penalty
                        if rnnlm:  # Word LM needs to add final <eos> score
                            hyp["score"] += recog_args.lm_weight * rnnlm.final(
                                hyp["rnnlm_prev"]
                            )
                        ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            # end detection
            if end_detect(ended_hyps, i) and recog_args.maxlenratio == 0.0:
                logging.info("end detected at %d", i)
                break

            hyps = remained_hyps
            if len(hyps) > 0:
                logging.debug("remeined hypothes: " + str(len(hyps)))
            else:
                logging.info("no hypothesis. Finish decoding.")
                break

            if char_list is not None:
                for hyp in hyps:
                    logging.debug(
                        "hypo: " + "".join([char_list[int(x)] for x in hyp["yseq"][1:]])
                    )

            logging.debug("number of ended hypothes: " + str(len(ended_hyps)))

        nbest_hyps = sorted(ended_hyps, key=lambda x: x["score"], reverse=True)[
            : min(len(ended_hyps), recog_args.nbest)
        ]

        # check number of hypotheis
        if len(nbest_hyps) == 0:
            logging.warning(
                "there is no N-best results, perform recognition "
                "again with smaller minlenratio."
            )
            # should copy becasuse Namespace will be overwritten globally
            recog_args = Namespace(**vars(recog_args))
            recog_args.minlenratio = max(0.0, recog_args.minlenratio - 0.1)
            return self.recognize(x, recog_args, char_list, rnnlm)

        logging.info("total log probability: " + str(nbest_hyps[0]["score"]))
        logging.info(
            "normalized log probability: "
            + str(nbest_hyps[0]["score"] / len(nbest_hyps[0]["yseq"]))
        )
        return nbest_hyps

    def recognize_maskctc(self, x, recog_args, char_list=None):
        """Non-autoregressive decoding using Mask CTC.

        :param ndnarray x: input acoustic feature (B, T, D) or (T, D)
        :param Namespace recog_args: argment Namespace contraining options
        :param list char_list: list of characters
        :return: decoding result
        :rtype: list
        """
        self.eval()
        h = self.encode(x).unsqueeze(0)

        ctc_probs, ctc_ids = torch.exp(self.ctc.log_softmax(h)).max(dim=-1)
        y_hat = torch.stack([x[0] for x in groupby(ctc_ids[0])])
        y_idx = torch.nonzero(y_hat != 0).squeeze(-1)

        probs_hat = []
        cnt = 0
        for i, y in enumerate(y_hat.tolist()):
            probs_hat.append(-1)
            while cnt < ctc_ids.shape[1] and y == ctc_ids[0][cnt]:
                if probs_hat[i] < ctc_probs[0][cnt]:
                    probs_hat[i] = ctc_probs[0][cnt].item()
                cnt += 1
        probs_hat = torch.from_numpy(numpy.array(probs_hat))

        char_mask = "_"
        p_thres = recog_args.maskctc_probability_threshold
        mask_idx = torch.nonzero(probs_hat[y_idx] < p_thres).squeeze(-1)
        confident_idx = torch.nonzero(probs_hat[y_idx] >= p_thres).squeeze(-1)
        mask_num = len(mask_idx)

        y_in = torch.zeros(1, len(y_idx) + 1, dtype=torch.long) + self.mask_token
        y_in[0][confident_idx] = y_hat[y_idx][confident_idx]
        y_in[0][-1] = self.eos

        logging.info(
            "ctc:{}".format(
                "".join(
                    [
                        char_list[y] if y != self.mask_token else char_mask
                        for y in y_in[0].tolist()
                    ]
                ).replace("<space>", " ")
            )
        )

        if not mask_num == 0:
            K = recog_args.maskctc_n_iterations
            num_iter = K if mask_num >= K and K > 0 else mask_num

            for t in range(1, num_iter):
                pred, _ = self.decoder(
                    y_in, (y_in != self.ignore_id).unsqueeze(-2), h, None
                )
                pred_sc, pred_id = pred[0][mask_idx].max(dim=-1)
                cand = torch.topk(pred_sc, mask_num // num_iter, -1)[1]
                y_in[0][mask_idx[cand]] = pred_id[cand]
                mask_idx = torch.nonzero(y_in[0] == self.mask_token).squeeze(-1)

                logging.info(
                    "msk:{}".format(
                        "".join(
                            [
                                char_list[y] if y != self.mask_token else char_mask
                                for y in y_in[0].tolist()
                            ]
                        ).replace("<space>", " ")
                    )
                )

            pred, pred_mask = self.decoder(
                y_in, (y_in != self.ignore_id).unsqueeze(-2), h, None
            )
            y_in[0][mask_idx] = pred[0][mask_idx].argmax(dim=-1)
            logging.info(
                "msk:{}".format(
                    "".join(
                        [
                            char_list[y] if y != self.mask_token else char_mask
                            for y in y_in[0].tolist()
                        ]
                    ).replace("<space>", " ")
                )
            )

        ret = y_in.tolist()[0][:-1]
        hyp = {"score": 0.0, "yseq": [self.sos] + ret + [self.eos]}

        return [hyp]

    def calculate_all_attentions(self, xs_pad, ilens, ys_pad):
        """E2E attention calculation.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :return: attention weights (B, H, Lmax, Tmax)
        :rtype: float ndarray
        """
        self.eval()
        with torch.no_grad():
            self.forward(xs_pad, ilens, ys_pad)
        ret = dict()
        for name, m in self.named_modules():
            if (
                isinstance(m, MultiHeadedAttention)
                or isinstance(m, DynamicConvolution)
                or isinstance(m, RelPositionMultiHeadedAttention)
            ):
                ret[name] = m.attn.cpu().numpy()
            if isinstance(m, DynamicConvolution2D):
                ret[name + "_time"] = m.attn_t.cpu().numpy()
                ret[name + "_freq"] = m.attn_f.cpu().numpy()
        self.train()
        return ret

    # GUO added
    def enhance(self, xs):
        """Forward only in the frontend stage.

        :param ndarray xs: input acoustic feature (T, C, F)
        :return: enhaned feature
        :rtype: torch.Tensor
        """
        if self.frontend is None:
            raise RuntimeError("Frontend does't exist")
        prev = self.training
        self.eval()
        ilens = np.fromiter((xx.shape[0] for xx in xs), dtype=np.int64)

        # subsample frame
        xs = [xx[:: self.subsample[0], :] for xx in xs]
        xs = [to_device(self, to_torch_tensor(xx).float()) for xx in xs]
        xs_pad = pad_list(xs, 0.0)
        enhanced, hlensm, mask = self.frontend(xs_pad, ilens)
        if prev:
            self.train()
        return enhanced.cpu().numpy(), mask.cpu().numpy(), ilens

    def calculate_all_ctc_probs(self, xs_pad, ilens, ys_pad):
        """E2E CTC probability calculation.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :return: CTC probability (B, Tmax, vocab)
        :rtype: float ndarray
        """
        ret = None
        if self.mtlalpha == 0:
            return ret

        self.eval()
        with torch.no_grad():
            self.forward(xs_pad, ilens, ys_pad)
        for name, m in self.named_modules():
            if isinstance(m, CTC) and m.probs is not None:
                ret = m.probs.cpu().numpy()
        self.train()
        return ret
    
    

    
