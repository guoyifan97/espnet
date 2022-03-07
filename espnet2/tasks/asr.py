import argparse
import logging
from sys import argv
from typing import Callable
from typing import Collection
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch
from torch.autograd import backward
from typeguard import check_argument_types
from typeguard import check_return_type
import itertools
from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.decoder.rnn_decoder import RNNDecoder
from espnet2.asr.decoder.transformer_decoder import (
    DynamicConvolution2DTransformerDecoder,  # noqa: H301
)
from espnet2.asr.decoder.transformer_decoder import DynamicConvolutionTransformerDecoder
from espnet2.asr.decoder.transformer_decoder import (
    LightweightConvolution2DTransformerDecoder,  # noqa: H301
)
from espnet2.asr.decoder.transformer_decoder import (
    LightweightConvolutionTransformerDecoder,  # noqa: H301
)
from espnet2.asr.decoder.transformer_decoder import TransformerDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.encoder.conformer_encoder import ConformerEncoder
from espnet2.asr.encoder.hubert_encoder import FairseqHubertEncoder
from espnet2.asr.encoder.hubert_encoder import FairseqHubertPretrainEncoder
from espnet2.asr.encoder.rnn_encoder import RNNEncoder
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.asr.encoder.contextual_block_transformer_encoder import (
    ContextualBlockTransformerEncoder,  # noqa: H301
)
from espnet2.asr.encoder.vgg_rnn_encoder import VGGRNNEncoder
from espnet2.asr.encoder.wav2vec2_encoder import FairSeqWav2Vec2Encoder
from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.asr.frontend.s3prl import S3prlFrontend
from espnet2.asr.frontend.windowing import SlidingWindow
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.postencoder.hugging_face_transformers_postencoder import (
    HuggingFaceTransformersPostEncoder,  # noqa: H301
)
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.preencoder.linear import LinearProjection
from espnet2.asr.preencoder.sinc import LightweightSincConvs
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asr.specaug.specaug import SpecAug
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.layers.global_mvn import GlobalMVN
from espnet2.layers.utterance_mvn import UtteranceMVN
from espnet2.tasks.abs_task import AbsTask
from espnet2.text.phoneme_tokenizer import g2p_choices
from espnet2.torch_utils.initialize import initialize
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.preprocessor import CommonPreprocessor, CommonPreprocessorMultiSpkrASR
from espnet2.train.trainer import Trainer
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import float_or_none
from espnet2.utils.types import int_or_none
from espnet2.utils.types import str2bool
from espnet2.utils.types import str_or_none

frontend_choices = ClassChoices(
    name="frontend",
    classes=dict(
        default=DefaultFrontend,
        sliding_window=SlidingWindow,
        s3prl=S3prlFrontend,
    ),
    type_check=AbsFrontend,
    default="default",
)
specaug_choices = ClassChoices(
    name="specaug",
    classes=dict(specaug=SpecAug),
    type_check=AbsSpecAug,
    default=None,
    optional=True,
)
normalize_choices = ClassChoices(
    "normalize",
    classes=dict(
        global_mvn=GlobalMVN,
        utterance_mvn=UtteranceMVN,
    ),
    type_check=AbsNormalize,
    default="utterance_mvn",
    optional=True,
)
preencoder_choices = ClassChoices(
    name="preencoder",
    classes=dict(
        sinc=LightweightSincConvs,
        linear=LinearProjection,
    ),
    type_check=AbsPreEncoder,
    default=None,
    optional=True,
)
encoder_choices = ClassChoices(
    "encoder",
    classes=dict(
        conformer=ConformerEncoder,
        transformer=TransformerEncoder,
        contextual_block_transformer=ContextualBlockTransformerEncoder,
        vgg_rnn=VGGRNNEncoder,
        rnn=RNNEncoder,
        wav2vec2=FairSeqWav2Vec2Encoder,
        hubert=FairseqHubertEncoder,
        hubert_pretrain=FairseqHubertPretrainEncoder,
    ),
    type_check=AbsEncoder,
    default="rnn",
)
postencoder_choices = ClassChoices(
    name="postencoder",
    classes=dict(
        hugging_face_transformers=HuggingFaceTransformersPostEncoder,
    ),
    type_check=AbsPostEncoder,
    default=None,
    optional=True,
)
decoder_choices = ClassChoices(
    "decoder",
    classes=dict(
        transformer=TransformerDecoder,
        lightweight_conv=LightweightConvolutionTransformerDecoder,
        lightweight_conv2d=LightweightConvolution2DTransformerDecoder,
        dynamic_conv=DynamicConvolutionTransformerDecoder,
        dynamic_conv2d=DynamicConvolution2DTransformerDecoder,
        rnn=RNNDecoder,
    ),
    type_check=AbsDecoder,
    default="rnn",
)

from espnet2.optimizers.sgd import SGD
optim_classes = dict(
    adam=torch.optim.Adam,
    adamw=torch.optim.AdamW,
    sgd=SGD,
    adadelta=torch.optim.Adadelta,
    adagrad=torch.optim.Adagrad,
    adamax=torch.optim.Adamax,
    asgd=torch.optim.ASGD,
    lbfgs=torch.optim.LBFGS,
    rmsprop=torch.optim.RMSprop,
    rprop=torch.optim.Rprop,
)
try:
    import torch_optimizer

    optim_classes.update(
        accagd=torch_optimizer.AccSGD,
        adabound=torch_optimizer.AdaBound,
        adamod=torch_optimizer.AdaMod,
        diffgrad=torch_optimizer.DiffGrad,
        lamb=torch_optimizer.Lamb,
        novograd=torch_optimizer.NovoGrad,
        pid=torch_optimizer.PID,
        # torch_optimizer<=0.0.1a10 doesn't support
        # qhadam=torch_optimizer.QHAdam,
        qhm=torch_optimizer.QHM,
        radam=torch_optimizer.RAdam,
        sgdw=torch_optimizer.SGDW,
        yogi=torch_optimizer.Yogi,
    )
    del torch_optimizer
except ImportError:
    pass
try:
    import apex

    optim_classes.update(
        fusedadam=apex.optimizers.FusedAdam,
        fusedlamb=apex.optimizers.FusedLAMB,
        fusednovograd=apex.optimizers.FusedNovoGrad,
        fusedsgd=apex.optimizers.FusedSGD,
    )
    del apex
except ImportError:
    pass
try:
    import fairscale
except ImportError:
    fairscale = None

class ASRTask(AbsTask):
    # If you need more than one optimizers, change this value
    num_optimizers: int = 2

    # Add variable objects configurations
    class_choices_list = [
        # --frontend and --frontend_conf
        frontend_choices,
        # --specaug and --specaug_conf
        specaug_choices,
        # --normalize and --normalize_conf
        normalize_choices,
        # --preencoder and --preencoder_conf
        preencoder_choices,
        # --encoder and --encoder_conf
        encoder_choices,
        # --postencoder and --postencoder_conf
        postencoder_choices,
        # --decoder and --decoder_conf
        decoder_choices,
    ]

    # If you need to modify train() or eval() procedures, change Trainer class here
    trainer = Trainer

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(description="Task related")

        # NOTE(kamo): add_arguments(..., required=True) can't be used
        # to provide --print_config mode. Instead of it, do as
        required = parser.get_default("required")
        required += ["token_list"]

        group.add_argument(
            "--token_list",
            type=str_or_none,
            default=None,
            help="A text mapping int-id to token",
        )
        group.add_argument(
            "--init",
            type=lambda x: str_or_none(x.lower()),
            default=None,
            help="The initialization method",
            choices=[
                "chainer",
                "xavier_uniform",
                "xavier_normal",
                "kaiming_uniform",
                "kaiming_normal",
                None,
            ],
        )

        group.add_argument(
            "--input_size",
            type=int_or_none,
            default=None,
            help="The number of input dimension of the feature",
        )

        group.add_argument(
            "--ctc_conf",
            action=NestedDictAction,
            default=get_default_kwargs(CTC),
            help="The keyword arguments for CTC class.",
        )
        group.add_argument(
            "--model_conf",
            action=NestedDictAction,
            default=get_default_kwargs(ESPnetASRModel),
            help="The keyword arguments for model class.",
        )

        group = parser.add_argument_group(description="Preprocess related")
        group.add_argument(
            "--use_preprocessor",
            type=str2bool,
            default=True,
            help="Apply preprocessing to data or not",
        )
        group.add_argument(
            "--use_preprocessor_multi_spkr",
            type=str2bool,
            default=False,
            help="whether use multi speaker preprocessor",
        )
        group.add_argument(
            "--token_type",
            type=str,
            default="bpe",
            choices=["bpe", "char", "word", "phn"],
            help="The text will be tokenized " "in the specified level token",
        )
        group.add_argument(
            "--bpemodel",
            type=str_or_none,
            default=None,
            help="The model file of sentencepiece",
        )
        parser.add_argument(
            "--non_linguistic_symbols",
            type=str_or_none,
            help="non_linguistic_symbols file path",
        )
        parser.add_argument(
            "--cleaner",
            type=str_or_none,
            choices=[None, "tacotron", "jaconv", "vietnamese"],
            default=None,
            help="Apply text cleaning",
        )
        parser.add_argument(
            "--g2p",
            type=str_or_none,
            choices=g2p_choices,
            default=None,
            help="Specify g2p method if --token_type=phn",
        )
        parser.add_argument(
            "--speech_volume_normalize",
            type=float_or_none,
            default=None,
            help="Scale the maximum amplitude to the given value.",
        )
        parser.add_argument(
            "--rir_scp",
            type=str_or_none,
            default=None,
            help="The file path of rir scp file.",
        )
        parser.add_argument(
            "--rir_apply_prob",
            type=float,
            default=1.0,
            help="THe probability for applying RIR convolution.",
        )
        parser.add_argument(
            "--noise_scp",
            type=str_or_none,
            default=None,
            help="The file path of noise scp file.",
        )
        parser.add_argument(
            "--noise_apply_prob",
            type=float,
            default=1.0,
            help="The probability applying Noise adding.",
        )
        parser.add_argument(
            "--noise_db_range",
            type=str,
            default="13_15",
            help="The range of noise decibel level.",
        )

        for class_choices in cls.class_choices_list:
            # Append --<name> and --<name>_conf.
            # e.g. --encoder and --encoder_conf
            class_choices.add_arguments(group)

    @classmethod
    def build_collate_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Callable[
        [Collection[Tuple[str, Dict[str, np.ndarray]]]],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:
        assert check_argument_types()
        # NOTE(kamo): int value = 0 is reserved by CTC-blank symbol
        return CommonCollateFn(float_pad_value=0.0, int_pad_value=-1)

    @classmethod
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        assert check_argument_types()
        if args.use_preprocessor:
            if not getattr(args, "use_preprocessor_multi_spkr", False):
                retval = CommonPreprocessor(
                    train=train,
                    token_type=args.token_type,
                    token_list=args.token_list,
                    bpemodel=args.bpemodel,
                    non_linguistic_symbols=args.non_linguistic_symbols,
                    text_cleaner=args.cleaner,
                    g2p_type=args.g2p,
                    # NOTE(kamo): Check attribute existence for backward compatibility
                    rir_scp=args.rir_scp if hasattr(args, "rir_scp") else None,
                    rir_apply_prob=args.rir_apply_prob
                    if hasattr(args, "rir_apply_prob")
                    else 1.0,
                    noise_scp=args.noise_scp if hasattr(args, "noise_scp") else None,
                    noise_apply_prob=args.noise_apply_prob
                    if hasattr(args, "noise_apply_prob")
                    else 1.0,
                    noise_db_range=args.noise_db_range
                    if hasattr(args, "noise_db_range")
                    else "13_15",
                    speech_volume_normalize=args.speech_volume_normalize
                    if hasattr(args, "rir_scp")
                    else None,
                )
            else:
                retval = CommonPreprocessorMultiSpkrASR(
                    train=train,
                    token_type=args.token_type,
                    token_list=args.token_list,
                    bpemodel=args.bpemodel,
                    non_linguistic_symbols=args.non_linguistic_symbols,
                    text_cleaner=args.cleaner,
                    g2p_type=args.g2p,
                    # NOTE(kamo): Check attribute existence for backward compatibility
                    rir_scp=args.rir_scp if hasattr(args, "rir_scp") else None,
                    rir_apply_prob=args.rir_apply_prob
                    if hasattr(args, "rir_apply_prob")
                    else 1.0,
                    noise_scp=args.noise_scp if hasattr(args, "noise_scp") else None,
                    noise_apply_prob=args.noise_apply_prob
                    if hasattr(args, "noise_apply_prob")
                    else 1.0,
                    noise_db_range=args.noise_db_range
                    if hasattr(args, "noise_db_range")
                    else "13_15",
                    speech_volume_normalize=args.speech_volume_normalize
                    if hasattr(args, "rir_scp")
                    else None,
                )
        else:
            retval = None
        assert check_return_type(retval)
        return retval

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        if not inference:
            retval = ("speech", "text")
        else:
            # Recognition mode
            retval = ("speech",)
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        retval = ()
        assert check_return_type(retval)
        return retval

    @classmethod
    def build_model(cls, args: argparse.Namespace) -> ESPnetASRModel:
        assert check_argument_types()
        if isinstance(args.token_list, str):
            with open(args.token_list, encoding="utf-8") as f:
                token_list = [line.rstrip() for line in f]

            # Overwriting token_list to keep it as "portable".
            args.token_list = list(token_list)
        elif isinstance(args.token_list, (tuple, list)):
            token_list = list(args.token_list)
        else:
            raise RuntimeError("token_list must be str or list")
        vocab_size = len(token_list)
        logging.info(f"Vocabulary size: {vocab_size }")

        # 1. frontend
        if args.input_size is None:
            # Extract features in the model
            # return torch model, for example DefaultFrontend: Guo
            frontend_class = frontend_choices.get_class(args.frontend)
            frontend = frontend_class(**args.frontend_conf)
            input_size = frontend.output_size()
        else:
            # Give features from data-loader
            args.frontend = None
            args.frontend_conf = {}
            frontend = None
            input_size = args.input_size

        # 2. Data augmentation for spectrogram
        if args.specaug is not None:
            specaug_class = specaug_choices.get_class(args.specaug)
            specaug = specaug_class(**args.specaug_conf)
        else:
            specaug = None

        # 3. Normalization layer
        if args.normalize is not None:
            normalize_class = normalize_choices.get_class(args.normalize)
            normalize = normalize_class(**args.normalize_conf)
        else:
            normalize = None

        # 4. Pre-encoder input block
        # NOTE(kan-bayashi): Use getattr to keep the compatibility
        if getattr(args, "preencoder", None) is not None:
            preencoder_class = preencoder_choices.get_class(args.preencoder)
            preencoder = preencoder_class(**args.preencoder_conf)
            input_size = preencoder.output_size()
        else:
            preencoder = None

        # 4. Encoder
        encoder_class = encoder_choices.get_class(args.encoder)
        encoder = encoder_class(input_size=input_size, **args.encoder_conf)

        # 5. Post-encoder block
        # NOTE(kan-bayashi): Use getattr to keep the compatibility
        encoder_output_size = encoder.output_size()
        if getattr(args, "postencoder", None) is not None:
            postencoder_class = postencoder_choices.get_class(args.postencoder)
            postencoder = postencoder_class(
                input_size=encoder_output_size, **args.postencoder_conf
            )
            encoder_output_size = postencoder.output_size()
        else:
            postencoder = None

        # 5. Decoder
        decoder_class = decoder_choices.get_class(args.decoder)

        decoder = decoder_class(
            vocab_size=vocab_size,
            encoder_output_size=encoder_output_size,
            **args.decoder_conf,
        )

        # 6. CTC
        ctc = CTC(
            odim=vocab_size, encoder_output_sizse=encoder_output_size, **args.ctc_conf
        )

        # 7. RNN-T Decoder (Not implemented)
        rnnt_decoder = None

        # 8. Build model
        model = ESPnetASRModel(
            vocab_size=vocab_size,
            frontend=frontend,
            specaug=specaug,
            normalize=normalize,
            preencoder=preencoder,
            encoder=encoder,
            postencoder=postencoder,
            decoder=decoder,
            ctc=ctc,
            rnnt_decoder=rnnt_decoder,
            token_list=token_list,
            **args.model_conf,
        )

        # FIXME(kamo): Should be done in model?
        # 9. Initialize
        if args.init is not None:
            initialize(model, args.init)

        assert check_return_type(model)
        return model
    
    @classmethod
    def build_optimizers(
        cls,
        args: argparse.Namespace,
        model: ESPnetASRModel,
    ) -> List[torch.optim.Optimizer]:

        # define generator optimizer
        optim_backend_class = optim_classes.get(args.optim)
        if optim_backend_class is None:
            raise ValueError(f"must be one of {list(optim_classes)}: {args.optim}")

        if "frontend_conf" in args.frontend_conf and "use_beamformer" in args.frontend_conf["frontend_conf"] and args.frontend_conf["frontend_conf"]["use_beamformer"] == True:
            logging.info(f"We use two optimizer for frontend and backend individually")
            backend_parameters_list = []

            if model.specaug != None:
                backend_parameters_list.append(model.specaug.parameters())
            
            if model.normalize != None:
                backend_parameters_list.append(model.normalize.parameters())
            
            if model.preencoder != None:
                backend_parameters_list.append(model.preencoder.parameters())
            
            if model.postencoder != None:
                backend_parameters_list.append(model.postencoder.parameters())
            
            if model.encoder != None:
                backend_parameters_list.append(model.encoder.parameters())
            
            if model.decoder != None:
                backend_parameters_list.append(model.decoder.parameters())
            
            if model.criterion_att != None:
                backend_parameters_list.append(model.criterion_att.parameters())
            
            if model.ctc != None:
                backend_parameters_list.append(model.ctc.parameters())
            
            backend_parameters = itertools.chain(*backend_parameters_list)
            frontend_parameters = model.frontend.parameters()
        else:
            backend_parameters = model.parameters()
            frontend_parameters = None
        
        if args.sharded_ddp:
            try:
                import fairscale
            except ImportError:
                raise RuntimeError("Requiring fairscale. Do 'pip install fairscale'")
            optim_backend = fairscale.optim.oss.OSS(
                params=backend_parameters,
                optim=optim_backend_class,
                **args.optim_conf,
            )
        else:
            optim_backend = optim_backend_class(
                backend_parameters,
                **args.optim_conf,
            )

        optimizers = [optim_backend]

        if frontend_parameters != None:
            optim_frontend_class = optim_classes.get(args.optim2)
            if optim_frontend_class is None:
                raise ValueError(f"must be one of {list(optim_classes)}: {args.optim2}")
            if args.sharded_ddp:
                try:
                    import fairscale
                except ImportError:
                    raise RuntimeError("Requiring fairscale. Do 'pip install fairscale'")
                optim_frontend = fairscale.optim.oss.OSS(
                    params=frontend_parameters,
                    optim=optim_frontend_class,
                    **args.optim2_conf,
                )
            else:
                optim_frontend = optim_frontend_class(
                    frontend_parameters,
                    **args.optim2_conf,
                )

            optimizers.append(optim_frontend) 

        return optimizers