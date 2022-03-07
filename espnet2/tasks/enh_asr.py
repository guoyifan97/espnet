import argparse
import copy
import logging
from typing import Callable
from typing import Collection
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch
from typeguard import check_argument_types
from typeguard import check_return_type

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.decoder.rnn_decoder import RNNDecoder
from espnet2.asr.decoder.transformer_decoder import TransformerDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.encoder.rnn_encoder import RNNEncoder
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.asr.encoder.vgg_rnn_encoder import VGGRNNEncoder
from espnet2.asr.espnet_enh_asr_model import ESPnetEnhASRModel
from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asr.specaug.specaug import SpecAug
from espnet2.enh.espnet_model import ESPnetEnhancementModel
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.layers.global_mvn import GlobalMVN
from espnet2.layers.utterance_mvn import UtteranceMVN
from espnet2.tasks.abs_task import AbsTask
from espnet2.tasks.enh import decoder_choices as enh_decoder_choices_
from espnet2.tasks.enh import encoder_choices as enh_encoder_choices_
from espnet2.tasks.enh import separator_choices as enh_separator_choices_
from espnet2.text.phoneme_tokenizer import g2p_choices
from espnet2.torch_utils.initialize import initialize
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.preprocessor import CommonPreprocessor_multi
from espnet2.train.trainer import Trainer
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import int_or_none
from espnet2.utils.types import str2bool
from espnet2.utils.types import str_or_none
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.postencoder.hugging_face_transformers_postencoder import (
    HuggingFaceTransformersPostEncoder,  # noqa: H301
)


# copy the choices from enh task to keep the choices consistent
# rename with "enh" prefix to avoid conficts with asr encoder/decoder
enh_encoder_choices = copy.deepcopy(enh_encoder_choices_)
enh_encoder_choices.name = "enh_encoder"
enh_decoder_choices = copy.deepcopy(enh_decoder_choices_)
enh_decoder_choices.name = "enh_decoder"
enh_separator_choices = copy.deepcopy(enh_separator_choices_)
enh_separator_choices.name = "enh_separator"

frontend_choices = ClassChoices(
    name="frontend",
    classes=dict(default=DefaultFrontend),
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
encoder_choices = ClassChoices(
    "encoder",
    classes=dict(
        transformer=TransformerEncoder,
        vgg_rnn=VGGRNNEncoder,
        rnn=RNNEncoder,
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
    classes=dict(transformer=TransformerDecoder, rnn=RNNDecoder),
    type_check=AbsDecoder,
    default="rnn",
)

MAX_REFERENCE_NUM = 100

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

class EnhASRTask(AbsTask):
    # If you need more than one optimizers, change this value
    num_optimizers: int = 2

    # Add variable objects configurations
    class_choices_list = [
        # --enh_encoder and --enh_encoder_conf
        enh_encoder_choices,
        # --enh_separator and --enh_separator_conf
        enh_separator_choices,
        # --enh_decoder and --enh_decoder_conf
        enh_decoder_choices,
        # --frontend and --frontend_conf
        frontend_choices,
        # --specaug and --specaug_conf
        specaug_choices,
        # --normalize and --normalize_conf
        normalize_choices,
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
            "--enh_model_conf",
            action=NestedDictAction,
            default=get_default_kwargs(ESPnetEnhancementModel),
            help="The keyword arguments for enh model class.",
        )
        group.add_argument(
            "--use_enh_model",
            type=str2bool,
            default=True,
            help="Apply preprocessing to data or not",
        )
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
            "--asr_model_conf",
            action=NestedDictAction,
            default=get_default_kwargs(ESPnetASRModel),
            help="The keyword arguments for ASR model class.",
        )

        group.add_argument(
            "--joint_model_conf",
            action=NestedDictAction,
            default=get_default_kwargs(ESPnetEnhASRModel),
            help="The keyword arguments for joint Enh and ASR model class.",
        )

        group = parser.add_argument_group(description="Preprocess related")
        group.add_argument(
            "--use_preprocessor",
            type=str2bool,
            default=False,
            help="Apply preprocessing to data or not",
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
        # Note(Jing): write the CommonPreprocessor_multi for multi args,
        # e.g., text_name = ["text_ref1" , "text_ref2"]
        # TODO(Jing): variable number of text_ref
        if args.use_preprocessor:
            retval = CommonPreprocessor_multi(
                train=train,
                token_type=args.token_type,
                token_list=args.token_list,
                bpemodel=args.bpemodel,
                non_linguistic_symbols=args.non_linguistic_symbols,
                text_name=["text_ref1", "text_ref2"],
                text_cleaner=args.cleaner,
                g2p_type=args.g2p,
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
            retval = ("speech_mix", "text_ref1")
        else:
            # Recognition mode
            retval = ("speech_mix",)
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        if not inference:
            retval = ["speech_ref{}".format(n) for n in range(1, MAX_REFERENCE_NUM + 1)]
            retval += ["text_ref{}".format(n) for n in range(2, MAX_REFERENCE_NUM + 1)]
            retval += ["noise_ref{}".format(n) for n in range(1, MAX_REFERENCE_NUM + 1)]
            retval += ["utt2category"]
            retval += [
                "dereverb_ref{}".format(n) for n in range(1, MAX_REFERENCE_NUM + 1)
            ]
        else:
            retval = ["speech_ref{}".format(n) for n in range(1, MAX_REFERENCE_NUM + 1)]
        retval = tuple(retval)
        assert check_return_type(retval)
        return retval

    @classmethod
    def build_model(cls, args: argparse.Namespace) -> ESPnetEnhASRModel:
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

        # 0. Build pre enhancement model
        if args.use_enh_model:
            enh_encoder = enh_encoder_choices.get_class(args.enh_encoder)(
                **args.enh_encoder_conf
            )
            enh_separator = enh_separator_choices.get_class(args.enh_separator)(
                enh_encoder.output_dim, **args.enh_separator_conf
            )
            enh_decoder = enh_decoder_choices.get_class(args.enh_decoder)(
                **args.enh_decoder_conf
            )
            enh_model = ESPnetEnhancementModel(
                encoder=enh_encoder,
                decoder=enh_decoder,
                separator=enh_separator,
                **args.enh_model_conf,
            )
        else:
            enh_model = None

        # Step 1-7 follows the asr.py to build asr_model.
        # 1. frontend
        if args.input_size is None:
            # Extract features in the model
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

        # 4. PreEncoder (Not implemented)
        pre_encoder = None

        # 5. Encoder
        encoder_class = encoder_choices.get_class(args.encoder)
        encoder = encoder_class(input_size=input_size, **args.encoder_conf)

        # 5.5 Post-encoder block
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

        # 6. Decoder
        decoder_class = decoder_choices.get_class(args.decoder)

        decoder = decoder_class(
            vocab_size=vocab_size,
            encoder_output_size=encoder.output_size(),
            **args.decoder_conf,
        )

        # 7. CTC
        ctc = CTC(
            odim=vocab_size, encoder_output_sizse=encoder.output_size(), **args.ctc_conf
        )

        # 8. RNN-T Decoder (Not implemented)
        rnnt_decoder = None

        asr_model = ESPnetASRModel(
            vocab_size=vocab_size,
            frontend=frontend,
            specaug=specaug,
            normalize=normalize,
            preencoder=pre_encoder,
            encoder=encoder,
            postencoder=postencoder,
            decoder=decoder,
            ctc=ctc,
            rnnt_decoder=rnnt_decoder,
            token_list=token_list,
            **args.asr_model_conf,
        )
        # 8. Build model
        model = ESPnetEnhASRModel(
            enh_model=enh_model,
            asr_model=asr_model,
            **args.joint_model_conf,
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
        model: ESPnetEnhASRModel,
    ) -> List[torch.optim.Optimizer]:

        # define generator optimizer
        optim_backend_class = optim_classes.get(args.optim)
        if optim_backend_class is None:
            raise ValueError(f"must be one of {list(optim_classes)}: {args.optim}")
        
        optim_frontend_class = optim_classes.get(args.optim2)
        if optim_frontend_class is None:
            raise ValueError(f"must be one of {list(optim_classes)}: {args.optim2}")

        if "frontend_conf" in args.frontend_conf and "use_beamformer" in args.frontend_conf["frontend_conf"] and args.frontend_conf["frontend_conf"]["use_beamformer"] == True:
            logging.info(f"We use two optimizer for enh_subclass + frontend and ASR backend individually")
            
            backend_parameters_list = []

            if model.asr_subclass.specaug != None:
                backend_parameters_list.append(model.asr_subclass.specaug.parameters())
            
            if model.asr_subclass.normalize != None:
                backend_parameters_list.append(model.asr_subclass.normalize.parameters())
            
            if model.asr_subclass.preencoder != None:
                backend_parameters_list.append(model.asr_subclass.preencoder.parameters())
            
            if model.asr_subclass.postencoder != None:
                backend_parameters_list.append(model.asr_subclass.postencoder.parameters())
            
            if model.asr_subclass.encoder != None:
                backend_parameters_list.append(model.asr_subclass.encoder.parameters())
            
            if model.asr_subclass.decoder != None:
                backend_parameters_list.append(model.asr_subclass.decoder.parameters())
            
            if model.asr_subclass.criterion_att != None:
                backend_parameters_list.append(model.asr_subclass.criterion_att.parameters())
            
            if model.asr_subclass.ctc != None:
                backend_parameters_list.append(model.asr_subclass.ctc.parameters())
            import itertools
            backend_parameters = itertools.chain(*backend_parameters_list)
            # frontend_parameters = itertools.chain(model.enh_subclass.parameters(), model.asr_subclass.frontend.parameters())
            frontend_parameters = itertools.chain(model.asr_subclass.frontend.parameters())
        else:
            logging.info(f"We use two optimizer for enh_subclass and asr_subclass backend individually")
            backend_parameters = model.asr_subclass.parameters()
            frontend_parameters = model.enh_subclass.parameters()
        
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
            optim_frontend = fairscale.optim.oss.OSS(
                params=frontend_parameters,
                optim=optim_frontend_class,
                **args.optim2_conf,
            )
        else:
            optim_backend = optim_backend_class(
                backend_parameters,
                **args.optim_conf,
            )
            optim_frontend = optim_frontend_class(
                frontend_parameters,
                **args.optim2_conf,
            )

        optimizers = [optim_backend, optim_frontend]

        return optimizers
