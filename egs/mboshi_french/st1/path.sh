MAIN_ROOT=$PWD/../../..
KALDI_ROOT=$MAIN_ROOT/tools/kaldi

export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PATH
export PATH=$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$MAIN_ROOT/tools/chainer_ctc/ext/warp-ctc/build
. "${MAIN_ROOT}"/tools/activate_python.sh && . "${MAIN_ROOT}"/tools/extra_path.sh
export PATH=$MAIN_ROOT/utils:$MAIN_ROOT/espnet/bin:$PATH

export OMP_NUM_THREADS=1

# check extra module installation
if ! which tokenizer.perl > /dev/null; then
    echo "Error: it seems that moses is not installed." >&2
    echo "Error: please install moses as follows." >&2
    echo "Error: cd ${MAIN_ROOT}/tools && make moses.done" >&2
    return 1
fi

# NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8
