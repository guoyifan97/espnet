import itertools
import logging

import numpy as np


def batchfy_by_seq(
    sorted_data,
    batch_size,
    max_length_in,
    max_length_out,
    min_batch_size=1,
    shortest_first=False,
    ikey="input",
    iaxis=0,
    okey="output",
    oaxis=0,
):
    """Make batch set from json dictionary

    :param Dict[str, Dict[str, Any]] sorted_data: dictionary loaded from data.json
    :param int batch_size: batch size
    :param int max_length_in: maximum length of input to decide adaptive batch size
    :param int max_length_out: maximum length of output to decide adaptive batch size
    :param int min_batch_size: mininum batch size (for multi-gpu)
    :param bool shortest_first: Sort from batch with shortest samples
        to longest if true, otherwise reverse
    :param str ikey: key to access input
        (for ASR ikey="input", for TTS, MT ikey="output".)
    :param int iaxis: dimension to access input
        (for ASR, TTS iaxis=0, for MT iaxis="1".)
    :param str okey: key to access output
        (for ASR, MT okey="output". for TTS okey="input".)
    :param int oaxis: dimension to access output
        (for ASR, TTS, MT oaxis=0, reserved for future research, -1 means all axis.)
    :return: List[List[Tuple[str, dict]]] list of batches
    """
    if batch_size <= 0:
        raise ValueError(f"Invalid batch_size={batch_size}")

    # check #utts is more than min_batch_size
    if len(sorted_data) < min_batch_size:
        raise ValueError(
            f"#utts({len(sorted_data)}) is less than min_batch_size({min_batch_size})."
        )

    # make list of minibatches
    minibatches = []
    start = 0
    while True:
        # [('7601-291468-0006_0', {'category': 'multichannel', 'input': [{'feat': '/home/guoyifan/MultiChannel/egs/libri_multi_baseline/data/dev_other/data/raw_pcm_dev_other.22.flac.h5:7601-291468-0006_0', 'filetype': 'sound.hdf5', 'name': 'input1', 'shape': [3516, 4, 257]}], 'output': [{'name': 'target1', 'shape': [99, 5002], 'text': 'HIS ABODE WHICH HE HAD FIXED AT A BOWERY OR COUNTRY SEAT AT A SHORT DISTANCE FROM THE CITY JUST AT WHAT IS NOW CALLED DUTCH STREET SOON ABOUNDED WITH PROOFS OF HIS INGENUITY PATENT SMOKE JACKS THAT REQUIRED A HORSE TO WORK THEM DUTCH OVENS THAT ROASTED MEAT WITHOUT FIRE CARTS THAT WENT BEFORE THE HORSES WEATHERCOCKS THAT TURNED AGAINST THE WIND AND OTHER WRONG HEADED CONTRIVANCES THAT ASTONISHED AND CONFOUNDED ALL BEHOLDERS', 'token': '▁HIS ▁A BO DE ▁WHICH ▁HE ▁HAD ▁FIXED ▁AT ▁A ▁BOW ERY ▁OR ▁COUNTRY ▁SEAT ▁AT ▁A ▁SHORT ▁DISTANCE ▁FROM ▁THE ▁CITY ▁JUST ▁AT ▁WHAT ▁IS ▁NOW ▁CALLED ▁DUTCH ▁STREET ▁SOON ▁A BOUND ED ▁WITH ▁PROOF S ▁OF ▁HIS ▁IN GEN U ITY ▁PAT ENT ▁SMOKE ▁JACK S ▁THAT ▁REQUIRED ▁A ▁HORSE ▁TO ▁WORK ▁THEM ▁DUTCH ▁ OV ENS ▁THAT ▁ROAST ED ▁ME AT ▁WITHOUT ▁FIRE ▁CAR T S ▁THAT ▁WENT ▁BEFORE ▁THE ▁HORSES ▁WEATHER CO CK S ▁THAT ▁TURNED ▁AGAINST ▁THE ▁WIND ▁AND ▁OTHER ▁WRONG ▁HEAD ED ▁CONTRIV ANCE S ▁THAT ▁ASTONISHED ▁AND ▁CONFOUND ED ▁ALL ▁BEHOLD ERS', 'tokenid': '2406 452 62 99 4885 2361 2318 2040 725 452 925 125 3280 1360 4002 725 452 4094 1618 2141 4537 1151 2692 725 4877 2630 3211 1008 1700 4357 4228 452 65 110 4931 3618 363 3247 2406 2523 153 404 223 3362 121 4191 2644 363 4536 3828 452 2437 4606 4952 4540 1700 451 324 120 4536 3892 110 2964 39 4935 2029 1028 382 363 4536 4870 821 4537 2439 4860 87 84 363 4536 4688 550 4537 4917 603 3293 4979 2362 110 1317 26 363 4536 722 603 1267 110 573 832 123'}], 'utt2spk': '7601-291468'}),...]

        _, info = sorted_data[start]
        ilen = int(info[ikey][iaxis]["shape"][0])
        olen = (
            int(info[okey][oaxis]["shape"][0])
            if oaxis >= 0
            else max(map(lambda x: int(x["shape"][0]), info[okey]))
        )
        factor = max(int(ilen / max_length_in), int(olen / max_length_out))
        # change batchsize depending on the input and output length
        # if ilen = 1000 and max_length_in = 800
        # then b = batchsize / 2
        # and max(min_batches, .) avoids batchsize = 0
        bs = max(min_batch_size, int(batch_size / (1 + factor)))
        end = min(len(sorted_data), start + bs)
        minibatch = sorted_data[start:end]
        if shortest_first:
            minibatch.reverse()

        # check each batch is more than minimum batchsize
        if len(minibatch) < min_batch_size:
            mod = min_batch_size - len(minibatch) % min_batch_size
            additional_minibatch = [
                sorted_data[i] for i in np.random.randint(0, start, mod)
            ]
            if shortest_first:
                additional_minibatch.reverse()
            minibatch.extend(additional_minibatch)
        minibatches.append(minibatch)

        if end == len(sorted_data):
            break
        start = end

    # batch: List[List[Tuple[str, dict]]]
    return minibatches


def batchfy_by_bin(
    sorted_data,
    batch_bins,
    num_batches=0,
    min_batch_size=1,
    shortest_first=False,
    ikey="input",
    okey="output",
):
    """Make variably sized batch set, which maximizes

    the number of bins up to `batch_bins`.

    :param List[Tuple[str, Dict[str, List[Dict[str, Any]]]] sorted_data: dictionary loaded from data.json
    :param int batch_bins: Maximum frames of a batch
    :param int num_batches: # number of batches to use (for debug)
    :param int min_batch_size: minimum batch size (for multi-gpu)
    :param int test: Return only every `test` batches
    :param bool shortest_first: Sort from batch with shortest samples
        to longest if true, otherwise reverse

    :param str ikey: key to access input (for ASR ikey="input", for TTS ikey="output".)
    :param str okey: key to access output (for ASR okey="output". for TTS okey="input".)

    :return: List[Tuple[str, Dict[str, List[Dict[str, Any]]]] list of batches
    """
    if batch_bins <= 0:
        raise ValueError(f"invalid batch_bins={batch_bins}")
    length = len(sorted_data)
    idim = int(sorted_data[0][1][ikey][0]["shape"][1])
    odim = int(sorted_data[0][1][okey][0]["shape"][1])
    logging.info("# utts: " + str(len(sorted_data)))
    minibatches = []
    start = 0
    n = 0
    while True:
        # Dynamic batch size depending on size of samples
        b = 0
        next_size = 0
        max_olen = 0
        while next_size < batch_bins and (start + b) < length:
            ilen = int(sorted_data[start + b][1][ikey][0]["shape"][0]) * idim
            olen = int(sorted_data[start + b][1][okey][0]["shape"][0]) * odim
            if olen > max_olen:
                max_olen = olen
            next_size = (max_olen + ilen) * (b + 1)
            if next_size <= batch_bins:
                b += 1
            elif next_size == 0:
                raise ValueError(
                    f"Can't fit one sample in batch_bins ({batch_bins}): "
                    f"Please increase the value"
                )
        end = min(length, start + max(min_batch_size, b))
        batch = sorted_data[start:end]
        if shortest_first:
            batch.reverse()
        minibatches.append(batch)
        # Check for min_batch_size and fixes the batches if needed
        i = -1
        while len(minibatches[i]) < min_batch_size:
            missing = min_batch_size - len(minibatches[i])
            if -i == len(minibatches):
                minibatches[i + 1].extend(minibatches[i])
                minibatches = minibatches[1:]
                break
            else:
                minibatches[i].extend(minibatches[i - 1][:missing])
                minibatches[i - 1] = minibatches[i - 1][missing:]
                i -= 1
        if end == length:
            break
        start = end
        n += 1
    if num_batches > 0:
        minibatches = minibatches[:num_batches]
    lengths = [len(x) for x in minibatches]
    logging.info(
        str(len(minibatches))
        + " batches containing from "
        + str(min(lengths))
        + " to "
        + str(max(lengths))
        + " samples "
        + "(avg "
        + str(int(np.mean(lengths)))
        + " samples)."
    )
    return minibatches


def batchfy_by_frame(
    sorted_data,
    max_frames_in,
    max_frames_out,
    max_frames_inout,
    num_batches=0,
    min_batch_size=1,
    shortest_first=False,
    ikey="input",
    okey="output",
):
    """Make variable batch set, which maximizes the number of frames to max_batch_frame.

    :param Dict[str, Dict[str, Any]] sorteddata: dictionary loaded from data.json
    :param int max_frames_in: Maximum input frames of a batch
    :param int max_frames_out: Maximum output frames of a batch
    :param int max_frames_inout: Maximum input+output frames of a batch
    :param int num_batches: # number of batches to use (for debug)
    :param int min_batch_size: minimum batch size (for multi-gpu)
    :param int test: Return only every `test` batches
    :param bool shortest_first: Sort from batch with shortest samples
        to longest if true, otherwise reverse

    :param str ikey: key to access input (for ASR ikey="input", for TTS ikey="output".)
    :param str okey: key to access output (for ASR okey="output". for TTS okey="input".)

    :return: List[Tuple[str, Dict[str, List[Dict[str, Any]]]] list of batches
    """
    if max_frames_in <= 0 and max_frames_out <= 0 and max_frames_inout <= 0:
        raise ValueError(
            "At least, one of `--batch-frames-in`, `--batch-frames-out` or "
            "`--batch-frames-inout` should be > 0"
        )
    length = len(sorted_data)
    minibatches = []
    start = 0
    end = 0
    while end != length:
        # Dynamic batch size depending on size of samples
        b = 0
        max_olen = 0
        max_ilen = 0
        while (start + b) < length:
            ilen = int(sorted_data[start + b][1][ikey][0]["shape"][0])
            if ilen > max_frames_in and max_frames_in != 0:
                raise ValueError(
                    f"Can't fit one sample in --batch-frames-in ({max_frames_in}): "
                    f"Please increase the value"
                )
            olen = int(sorted_data[start + b][1][okey][0]["shape"][0])
            if olen > max_frames_out and max_frames_out != 0:
                raise ValueError(
                    f"Can't fit one sample in --batch-frames-out ({max_frames_out}): "
                    f"Please increase the value"
                )
            if ilen + olen > max_frames_inout and max_frames_inout != 0:
                raise ValueError(
                    f"Can't fit one sample in --batch-frames-out ({max_frames_inout}): "
                    f"Please increase the value"
                )
            max_olen = max(max_olen, olen)
            max_ilen = max(max_ilen, ilen)
            in_ok = max_ilen * (b + 1) <= max_frames_in or max_frames_in == 0
            out_ok = max_olen * (b + 1) <= max_frames_out or max_frames_out == 0
            inout_ok = (max_ilen + max_olen) * (
                b + 1
            ) <= max_frames_inout or max_frames_inout == 0
            if in_ok and out_ok and inout_ok:
                # add more seq in the minibatch
                b += 1
            else:
                # no more seq in the minibatch
                break
        end = min(length, start + b)
        batch = sorted_data[start:end]
        if shortest_first:
            batch.reverse()
        minibatches.append(batch)
        # Check for min_batch_size and fixes the batches if needed
        i = -1
        while len(minibatches[i]) < min_batch_size:
            missing = min_batch_size - len(minibatches[i])
            if -i == len(minibatches):
                minibatches[i + 1].extend(minibatches[i])
                minibatches = minibatches[1:]
                break
            else:
                minibatches[i].extend(minibatches[i - 1][:missing])
                minibatches[i - 1] = minibatches[i - 1][missing:]
                i -= 1
        start = end
    if num_batches > 0:
        minibatches = minibatches[:num_batches]
    lengths = [len(x) for x in minibatches]
    logging.info(
        str(len(minibatches))
        + " batches containing from "
        + str(min(lengths))
        + " to "
        + str(max(lengths))
        + " samples"
        + "(avg "
        + str(int(np.mean(lengths)))
        + " samples)."
    )

    return minibatches


def batchfy_shuffle(data, batch_size, min_batch_size, num_batches, shortest_first):
    import random

    logging.info("use shuffled batch.")
    sorted_data = random.sample(data.items(), len(data.items()))
    logging.info("# utts: " + str(len(sorted_data)))
    # make list of minibatches
    minibatches = []
    start = 0
    while True:
        end = min(len(sorted_data), start + batch_size)
        # check each batch is more than minimum batchsize
        minibatch = sorted_data[start:end]
        if shortest_first:
            minibatch.reverse()
        if len(minibatch) < min_batch_size:
            mod = min_batch_size - len(minibatch) % min_batch_size
            additional_minibatch = [
                sorted_data[i] for i in np.random.randint(0, start, mod)
            ]
            if shortest_first:
                additional_minibatch.reverse()
            minibatch.extend(additional_minibatch)
        minibatches.append(minibatch)
        if end == len(sorted_data):
            break
        start = end

    # for debugging
    if num_batches > 0:
        minibatches = minibatches[:num_batches]
        logging.info("# minibatches: " + str(len(minibatches)))
    return minibatches


BATCH_COUNT_CHOICES = ["auto", "seq", "bin", "frame"]
BATCH_SORT_KEY_CHOICES = ["input", "output", "shuffle"]


def make_batchset(
    data,
    batch_size=0,
    max_length_in=float("inf"),
    max_length_out=float("inf"),
    num_batches=0,
    min_batch_size=1,
    shortest_first=False,
    batch_sort_key="input",
    swap_io=False,
    mt=False,
    count="auto",
    batch_bins=0,
    batch_frames_in=0,
    batch_frames_out=0,
    batch_frames_inout=0,
    iaxis=0,
    oaxis=0,
):
    """Make batch set from json dictionary

    if utts have "category" value,

        >>> data = {'utt1': {'category': 'A', 'input': ...},
        ...         'utt2': {'category': 'B', 'input': ...},
        ...         'utt3': {'category': 'B', 'input': ...},
        ...         'utt4': {'category': 'A', 'input': ...}}
        >>> make_batchset(data, batchsize=2, ...)
        [[('utt1', ...), ('utt4', ...)], [('utt2', ...), ('utt3': ...)]]

    Note that if any utts doesn't have "category",
    perform as same as batchfy_by_{count}

    :param Dict[str, Dict[str, Any]] data: dictionary loaded from data.json
    :param int batch_size: maximum number of sequences in a minibatch.
    :param int batch_bins: maximum number of bins (frames x dim) in a minibatch.
    :param int batch_frames_in:  maximum number of input frames in a minibatch.
    :param int batch_frames_out: maximum number of output frames in a minibatch.
    :param int batch_frames_out: maximum number of input+output frames in a minibatch.
    :param str count: strategy to count maximum size of batch.
        For choices, see espnet.asr.batchfy.BATCH_COUNT_CHOICES

    :param int max_length_in: maximum length of input to decide adaptive batch size
    :param int max_length_out: maximum length of output to decide adaptive batch size
    :param int num_batches: # number of batches to use (for debug)
    :param int min_batch_size: minimum batch size (for multi-gpu)
    :param bool shortest_first: Sort from batch with shortest samples
        to longest if true, otherwise reverse
    :param str batch_sort_key: how to sort data before creating minibatches
        ["input", "output", "shuffle"]
    :param bool swap_io: if True, use "input" as output and "output"
        as input in `data` dict
    :param bool mt: if True, use 0-axis of "output" as output and 1-axis of "output"
        as input in `data` dict
    :param int iaxis: dimension to access input
        (for ASR, TTS iaxis=0, for MT iaxis="1".)
    :param int oaxis: dimension to access output (for ASR, TTS, MT oaxis=0,
        reserved for future research, -1 means all axis.)
    :return: List[List[Tuple[str, dict]]] list of batches
    """

    # check args
    if count not in BATCH_COUNT_CHOICES:
        raise ValueError(
            f"arg 'count' ({count}) should be one of {BATCH_COUNT_CHOICES}"
        )
    if batch_sort_key not in BATCH_SORT_KEY_CHOICES:
        raise ValueError(
            f"arg 'batch_sort_key' ({batch_sort_key}) should be "
            f"one of {BATCH_SORT_KEY_CHOICES}"
        )

    # TODO(karita): remove this by creating converter from ASR to TTS json format
    batch_sort_axis = 0
    if swap_io:
        # for TTS
        ikey = "output"
        okey = "input"
        if batch_sort_key == "input":
            batch_sort_key = "output"
        elif batch_sort_key == "output":
            batch_sort_key = "input"
    elif mt:
        # for MT
        ikey = "output"
        okey = "output"
        batch_sort_key = "output"
        batch_sort_axis = 1
        assert iaxis == 1
        assert oaxis == 0
        # NOTE: input is json['output'][1] and output is json['output'][0]
    else:
        ikey = "input"
        okey = "output"

    if count == "auto":
        if batch_size != 0:
            count = "seq"
        elif batch_bins != 0:
            count = "bin"
        elif batch_frames_in != 0 or batch_frames_out != 0 or batch_frames_inout != 0:
            count = "frame"
        else:
            raise ValueError(
                f"cannot detect `count` manually set one of {BATCH_COUNT_CHOICES}"
            )
        logging.info(f"count is auto detected as {count}")

    if count != "seq" and batch_sort_key == "shuffle":
        raise ValueError("batch_sort_key=shuffle is only available if batch_count=seq")

    category2data = {}  # Dict[str, dict]
    for k, v in data.items():
        category2data.setdefault(v.get("category"), {})[k] = v

    batches_list = []  # List[List[List[Tuple[str, dict]]]]
    for d in category2data.values(): # d is dict; "multichannel": {}, "None": {}
        if batch_sort_key == "shuffle":
            batches = batchfy_shuffle(
                d, batch_size, min_batch_size, num_batches, shortest_first
            )
            batches_list.append(batches)
            continue

        # sort it by input lengths (long to short) 
        # 'input': [{'feat': '/home/guoyifan/MultiChannel/egs/libri_multi_baseline/data/dev_other/data/raw_pcm_dev_other.1.flac.h5:116-288045-0000_0', 'filetype': 'sound.hdf5', 'name': 'input1', 'shape': [1066, 4, 257]}],
        # d -> 
        # {
        # "116-288045-0000_0": {
        #     "category": "multichannel",
        #     "input": [
        #         {
        #             "feat": "/home/guoyifan/MultiChannel/egs/libri_multi_baseline/data/dev_other/data/raw_pcm_dev_other.1.flac.h5:116-288045-0000_0",
        #             "filetype": "sound.hdf5",
        #             "name": "input1",
        #             "shape": [
        #                 1066,
        #                 4,
        #                 257
        #             ]
        #         }
        #     ],
        #     "output": [
        #         {
        #             "name": "target1",
        #             "shape": [
        #                 43,
        #                 5002
        #             ],
        #             "text": "AS I APPROACHED THE CITY I HEARD BELLS RINGING AND A LITTLE LATER I FOUND THE STREETS ASTIR WITH THRONGS OF WELL DRESSED PEOPLE IN FAMILY GROUPS WENDING THEIR WAY HITHER AND THITHER",
        #             "token": "▁AS ▁I ▁APPROACHED ▁THE ▁CITY ▁I ▁HEARD ▁BELL S ▁RING ING ▁AND ▁A ▁LITTLE ▁LATER ▁I ▁FOUND ▁THE ▁STREETS ▁AS T IR ▁WITH ▁THRONG S ▁OF ▁WELL ▁DRESSED ▁PEOPLE ▁IN ▁FAMILY ▁GROUP S ▁WE N D ING ▁THEIR ▁WAY ▁HIT HER ▁AND ▁THITHER",
        #             "tokenid": "698 2484 660 4537 1151 2484 2368 837 363 3882 203 603 452 2841 2760 2484 2109 4537 4358 698 382 209 4931 4584 363 3247 4868 1671 3393 2523 1950 2294 363 4854 283 96 203 4539 4852 2410 166 603 4562"
        #         }
        #     ],
        #     "utt2spk": "116-288045"
        #   },
        # "116-288045-0000_1": {
        #     "category": "multichannel",
        #     "input": [
        #         {
        #             "feat": "/home/guoyifan/MultiChannel/egs/libri_multi_baseline/data/dev_other/data/raw_pcm_dev_other.1.flac.h5:116-288045-0000_1",
        #             "filetype": "sound.hdf5",
        #             "name": "input1",
        #             "shape": [
        #                 1066,
        #                 4,
        #                 257
        #             ]
        #         }
        #     ],
        #     "output": [
        #         {
        #             "name": "target1",
        #             "shape": [
        #                 43,
        #                 5002
        #             ],
        #             "text": "AS I APPROACHED THE CITY I HEARD BELLS RINGING AND A LITTLE LATER I FOUND THE STREETS ASTIR WITH THRONGS OF WELL DRESSED PEOPLE IN FAMILY GROUPS WENDING THEIR WAY HITHER AND THITHER",
        #             "token": "▁AS ▁I ▁APPROACHED ▁THE ▁CITY ▁I ▁HEARD ▁BELL S ▁RING ING ▁AND ▁A ▁LITTLE ▁LATER ▁I ▁FOUND ▁THE ▁STREETS ▁AS T IR ▁WITH ▁THRONG S ▁OF ▁WELL ▁DRESSED ▁PEOPLE ▁IN ▁FAMILY ▁GROUP S ▁WE N D ING ▁THEIR ▁WAY ▁HIT HER ▁AND ▁THITHER",
        #             "tokenid": "698 2484 660 4537 1151 2484 2368 837 363 3882 203 603 452 2841 2760 2484 2109 4537 4358 698 382 209 4931 4584 363 3247 4868 1671 3393 2523 1950 2294 363 4854 283 96 203 4539 4852 2410 166 603 4562"
        #         }
        #     ],
        #     "utt2spk": "116-288045"
        #   },
        # }
        sorted_data = sorted(
            d.items(),
            key=lambda data: int(data[1][batch_sort_key][batch_sort_axis]["shape"][0]),
            reverse=not shortest_first,
        )
        # sorted_data -> List(Tuple(str, Dict))
        # [('7601-291468-0006_0', {'category': 'multichannel', 'input': [{'feat': '/home/guoyifan/MultiChannel/egs/libri_multi_baseline/data/dev_other/data/raw_pcm_dev_other.22.flac.h5:7601-291468-0006_0', 'filetype': 'sound.hdf5', 'name': 'input1', 'shape': [3516, 4, 257]}], 'output': [{'name': 'target1', 'shape': [99, 5002], 'text': 'HIS ABODE WHICH HE HAD FIXED AT A BOWERY OR COUNTRY SEAT AT A SHORT DISTANCE FROM THE CITY JUST AT WHAT IS NOW CALLED DUTCH STREET SOON ABOUNDED WITH PROOFS OF HIS INGENUITY PATENT SMOKE JACKS THAT REQUIRED A HORSE TO WORK THEM DUTCH OVENS THAT ROASTED MEAT WITHOUT FIRE CARTS THAT WENT BEFORE THE HORSES WEATHERCOCKS THAT TURNED AGAINST THE WIND AND OTHER WRONG HEADED CONTRIVANCES THAT ASTONISHED AND CONFOUNDED ALL BEHOLDERS', 'token': '▁HIS ▁A BO DE ▁WHICH ▁HE ▁HAD ▁FIXED ▁AT ▁A ▁BOW ERY ▁OR ▁COUNTRY ▁SEAT ▁AT ▁A ▁SHORT ▁DISTANCE ▁FROM ▁THE ▁CITY ▁JUST ▁AT ▁WHAT ▁IS ▁NOW ▁CALLED ▁DUTCH ▁STREET ▁SOON ▁A BOUND ED ▁WITH ▁PROOF S ▁OF ▁HIS ▁IN GEN U ITY ▁PAT ENT ▁SMOKE ▁JACK S ▁THAT ▁REQUIRED ▁A ▁HORSE ▁TO ▁WORK ▁THEM ▁DUTCH ▁ OV ENS ▁THAT ▁ROAST ED ▁ME AT ▁WITHOUT ▁FIRE ▁CAR T S ▁THAT ▁WENT ▁BEFORE ▁THE ▁HORSES ▁WEATHER CO CK S ▁THAT ▁TURNED ▁AGAINST ▁THE ▁WIND ▁AND ▁OTHER ▁WRONG ▁HEAD ED ▁CONTRIV ANCE S ▁THAT ▁ASTONISHED ▁AND ▁CONFOUND ED ▁ALL ▁BEHOLD ERS', 'tokenid': '2406 452 62 99 4885 2361 2318 2040 725 452 925 125 3280 1360 4002 725 452 4094 1618 2141 4537 1151 2692 725 4877 2630 3211 1008 1700 4357 4228 452 65 110 4931 3618 363 3247 2406 2523 153 404 223 3362 121 4191 2644 363 4536 3828 452 2437 4606 4952 4540 1700 451 324 120 4536 3892 110 2964 39 4935 2029 1028 382 363 4536 4870 821 4537 2439 4860 87 84 363 4536 4688 550 4537 4917 603 3293 4979 2362 110 1317 26 363 4536 722 603 1267 110 573 832 123'}], 'utt2spk': '7601-291468'}),...]
        logging.info("# utts: " + str(len(sorted_data)))
        if count == "seq":
            batches = batchfy_by_seq(
                sorted_data,
                batch_size=batch_size,
                max_length_in=max_length_in,
                max_length_out=max_length_out,
                min_batch_size=min_batch_size,
                shortest_first=shortest_first,
                ikey=ikey,
                iaxis=iaxis,
                okey=okey,
                oaxis=oaxis,
            )
        if count == "bin":
            batches = batchfy_by_bin(
                sorted_data,
                batch_bins=batch_bins,
                min_batch_size=min_batch_size,
                shortest_first=shortest_first,
                ikey=ikey,
                okey=okey,
            )
        if count == "frame":
            batches = batchfy_by_frame(
                sorted_data,
                max_frames_in=batch_frames_in,
                max_frames_out=batch_frames_out,
                max_frames_inout=batch_frames_inout,
                min_batch_size=min_batch_size,
                shortest_first=shortest_first,
                ikey=ikey,
                okey=okey,
            )
        batches_list.append(batches)

    if len(batches_list) == 1:
        batches = batches_list[0]
    else:
        # Concat list. This way is faster than "sum(batch_list, [])"
        batches = list(itertools.chain(*batches_list))

    # for debugging
    if num_batches > 0:
        batches = batches[:num_batches]
    logging.info("# minibatches: " + str(len(batches)))

    # batch: List(num_batches)[[Tuple[str(uttid), dict(json,feat,name,shape...)]],[Tuple[str(uttid), dict(json,feat,name,shape...)]], ...]
    return batches
