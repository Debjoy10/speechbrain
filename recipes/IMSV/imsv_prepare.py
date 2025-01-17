"""
Data preparation - I-MSV 2022: Indic- Multilingual Speaker Verification Challenge 2022

Download: https://vlsp.org.vn/cocosda2022/i-msv
"""

import os
import csv
import logging
import glob
import random
import shutil
import sys  # noqa F401
import numpy as np
import torch
import torchaudio
from tqdm.contrib import tqdm
from speechbrain.dataio.dataio import (
    load_pkl,
    save_pkl,
)

logger = logging.getLogger(__name__)
OPT_FILE = "opt_imsv_prepare.pkl"
TRAIN_CSV = "train.csv"
DEV_CSV = "dev.csv"
TEST_CSV = "test.csv"
ENROL_CSV = "enrol.csv"
HIDDEN_TEST_CSV = "hidden_test.csv"
HIDDEN_ENROL_CSV = "hidden_enrol.csv"
SAMPLERATE = 16000
all_langs = ['EN', 'KN', 'HN', 'ML', 'TL', 'BN', 'AS', 'OR', 'MZ', 'TM', 'GJ', 'MR']

def prepare_imsv(
    data_folder,
    save_folder,
    verification_pairs_file,
    test_pairs_file,
    hidden_test_pairs_file,
    splits=["train", "dev", "test"],
    split_ratio=[90, 10], # Train - Dev Split for PLDA only.
    seg_dur=3.0,
    amp_th=5e-04,
    split_speaker=False,
    random_segment=False,
    skip_prep=False,
    langs=None,
):
    """
    Prepares the csv files for the I-MSV datasets.
    NOTE: 10 % of the training dataset (IMSV-Dev) can be split to use for validation

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original VoxCeleb dataset is stored.
    save_folder : str
        The directory where to store the csv files.
    verification_pairs_file : str
        txt file containing the verification split.
    splits : list
        List of splits to prepare from ['train', 'dev']
    split_ratio : list
        List if int for train and validation splits
    seg_dur : int
        Segment duration of a chunk in seconds (e.g., 3.0 seconds).
    amp_th : float
        removes segments whose average amplitude is below the
        given threshold.
    split_speaker : bool
        Speaker-wise split
    random_segment : bool
        Train random segments
    skip_prep: Bool
        If True, skip preparation.

    Example
    -------
    >>> from recipes.IMSV.imsv_prepare import prepare_imsv
    >>> data_folder = 'data/IMSV/'
    >>> save_folder = 'IMSV_OUT/'
    >>> splits = ['train', 'dev']
    >>> split_ratio = [90, 10]
    >>> prepare_voxceleb(data_folder, save_folder, splits, split_ratio)
    """

    if skip_prep:
        return
    # Create configuration for easily skipping data_preparation stage
    conf = {
        "data_folder": data_folder,
        "splits": splits,
        "split_ratio": split_ratio,
        "save_folder": save_folder,
        "seg_dur": seg_dur,
        "split_speaker": split_speaker,
    }

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Setting ouput files
    save_opt = os.path.join(save_folder, OPT_FILE)
    save_csv_train = os.path.join(save_folder, TRAIN_CSV)
    save_csv_dev = os.path.join(save_folder, DEV_CSV)

    # Check if this phase is already done (if so, skip it)
    if skip(splits, save_folder, conf):
        logger.info("Skipping preparation, completed in previous run.")
        return

    # Additional checks to make sure the data folder contains VoxCeleb data
    if "," in data_folder:
        data_folder = data_folder.replace(" ", "").split(",")
    else:
        data_folder = [data_folder]

    # _check_voxceleb1_folders(data_folder, splits)

    msg = "\tCreating csv file for the I-MSV Dataset.."
    logger.info(msg)
    
    # Checking the language to train on
    global all_langs
    if langs is not None:
        all_langs = langs

    # Split data into 90% train and 10% validation (verification split)
    wav_lst_train, wav_lst_dev = _get_utt_split_lists(
        data_folder, split_ratio, split_speaker
    )

    # Creating csv file for training data
    if "train" in splits:
        prepare_csv(
            seg_dur, wav_lst_train, save_csv_train, random_segment, amp_th
        )

    if "dev" in splits:
        prepare_csv(seg_dur, wav_lst_dev, save_csv_dev, random_segment, amp_th)

    if "enr" in splits:
        prepare_csv_enrol_test(
            data_folder, save_folder, verification_pairs_file
        )

    # For verification
    if "test" in splits:
        prepare_csv_enrol_test(
            data_folder, save_folder, test_pairs_file
        )

    # For testing
    if "hidden_test" in splits:
        prepare_csv_enrol_test_hidden(
            data_folder, save_folder, hidden_test_pairs_file
        )

    # Saving options (useful to skip this phase when already done)
    save_pkl(conf, save_opt)


def skip(splits, save_folder, conf):
    """
    Detects if the voxceleb data_preparation has been already done.
    If the preparation has been done, we can skip it.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    # Checking csv files
    skip = True

    split_files = {
        "train": TRAIN_CSV,
        "dev": DEV_CSV,
        "test": TEST_CSV,
        "enr": ENROL_CSV,
        "hidden_test": HIDDEN_TEST_CSV,
        "hidden_enr": HIDDEN_ENROL_CSV
    }

    for split in splits:
        if split not in split_files or not os.path.isfile(os.path.join(save_folder, split_files[split])):
            skip = False
    #  Checking saved options
    save_opt = os.path.join(save_folder, OPT_FILE)
    if skip is True:
        if os.path.isfile(save_opt):
            opts_old = load_pkl(save_opt)
            if opts_old == conf:
                skip = True
            else:
                skip = False
        else:
            skip = False

    return skip


# Used for verification split
def _get_utt_split_lists(
    data_folders, split_ratio, split_speaker=False
):
    """
    Tot. number of speakers IMSV-Dev = 50.
    Tot. number of speakers IMSV-Enr = 50.
    Splits the audio file list into train and dev.
    This function automatically removes verification test files from the training and dev set (if any).
    """
    train_lst = []
    dev_lst = []
    global all_langs
    print("Training with the following languages = {}".format(all_langs))

    print("Getting file list...")
    for data_folder in data_folders:

        path = os.path.join(data_folder, "I_MSV_DEV_ENR/Dev_data/", "*.wav")

        if split_speaker:
            # Speakers present in train and dev splits disjoint sets - For metric learning
            audio_files_dict = {}
            for f in glob.glob(path, recursive=True):
                spk_id = os.path.basename(f).split("_")[0]
                if f[-7:-5] not in all_langs:
                    continue
                audio_files_dict.setdefault(spk_id, []).append(f)

            spk_id_list = list(audio_files_dict.keys())
            random.shuffle(spk_id_list)
            split = int(0.01 * split_ratio[0] * len(spk_id_list))
            for spk_id in spk_id_list[:split]:
                train_lst.extend(audio_files_dict[spk_id])

            for spk_id in spk_id_list[split:]:
                dev_lst.extend(audio_files_dict[spk_id])
        else:
            # For speaker classification training - Using
            audio_files_list = []
            for f in glob.glob(path, recursive=True):
                try:
                    spk_id = os.path.basename(f).split("_")[0]
                    if f[-7:-5] not in all_langs:
                        continue
                except ValueError:
                    logger.info(f"Malformed path: {f}")
                    continue
                audio_files_list.append(f)

            random.shuffle(audio_files_list)
            split = int(0.01 * split_ratio[0] * len(audio_files_list))
            train_snts = audio_files_list[:split]
            dev_snts = audio_files_list[split:]

            train_lst.extend(train_snts)
            dev_lst.extend(dev_snts)

    return train_lst, dev_lst


def _get_chunks(seg_dur, audio_id, audio_duration):
    """
    Returns list of chunks
    """
    num_chunks = int(audio_duration / seg_dur)  # all in milliseconds

    chunk_lst = [
        audio_id + "_" + str(i * seg_dur) + "_" + str(i * seg_dur + seg_dur)
        for i in range(num_chunks)
    ]

    return chunk_lst


def prepare_csv(seg_dur, wav_lst, csv_file, random_segment=False, amp_th=0):
    """
    Creates the csv file given a list of wav files.

    Arguments
    ---------
    wav_lst : list
        The list of wav files of a given data split.
    csv_file : str
        The path of the output csv file
    random_segment: bool
        Read random segments
    amp_th: float
        Threshold on the average amplitude on the chunk.
        If under this threshold, the chunk is discarded.

    Returns
    -------
    None
    """

    msg = '\t"Creating csv lists in  %s..."' % (csv_file)
    logger.info(msg)

    csv_output = [["ID", "duration", "wav", "start", "stop", "spk_id"]]

    # For assigning unique ID to each chunk
    my_sep = "--"
    entry = []
    # Processing all the wav files in the list
    for wav_file in tqdm(wav_lst, dynamic_ncols=True):
        # Getting sentence and speaker ids
        try:
            wav_name = os.path.basename(wav_file).strip('.wav')
            [spk_id, sess_id, device_id, env_id, lang_id, style_id] = [wav_name[:4], wav_name[5], wav_name[6:9], wav_name[9], wav_name[10:12], wav_name[12]]
        except ValueError:
            logger.info(f"Malformed path: {wav_file}")
            continue
        audio_id = wav_name

        # Reading the signal (to retrieve duration in seconds)
        signal, fs = torchaudio.load(wav_file)
        signal = signal.squeeze(0)

        if random_segment:
            audio_duration = signal.shape[0] / SAMPLERATE
            start_sample = 0
            stop_sample = signal.shape[0]

            # Composition of the csv_line
            csv_line = [
                audio_id,
                str(audio_duration),
                wav_file,
                start_sample,
                stop_sample,
                spk_id,
            ]
            entry.append(csv_line)
        else:
            audio_duration = signal.shape[0] / SAMPLERATE

            uniq_chunks_list = _get_chunks(seg_dur, audio_id, audio_duration)
            for chunk in uniq_chunks_list:
                s, e = chunk.split("_")[-2:]
                start_sample = int(float(s) * SAMPLERATE)
                end_sample = int(float(e) * SAMPLERATE)

                #  Avoid chunks with very small energy
                mean_sig = torch.mean(np.abs(signal[start_sample:end_sample]))
                if mean_sig < amp_th:
                    continue

                # Composition of the csv_line
                csv_line = [
                    chunk,
                    str(audio_duration),
                    wav_file,
                    start_sample,
                    end_sample,
                    spk_id,
                ]
                entry.append(csv_line)

    csv_output = csv_output + entry

    # Writing the csv lines
    with open(csv_file, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        for line in csv_output:
            csv_writer.writerow(line)

    # Final prints
    msg = "\t%s successfully created!" % (csv_file)
    logger.info(msg)


def prepare_csv_enrol_test(data_folders, save_folder, verification_pairs_file):
    """
    Creates the csv file for test data (useful for verification)

    Arguments
    ---------
    data_folder : str
        Path of the data folders
    save_folder : str
        The directory where to store the csv files.

    Returns
    -------
    None
    """

    csv_output_head = [
        ["ID", "duration", "wav", "start", "stop", "spk_id"]
    ]  # noqa E231

    for data_folder in data_folders:

        test_lst_file = verification_pairs_file

        enrol_files, test_files = [], []

        # Get unique ids (enrol and test utterances)
        for line in open(test_lst_file):
            e_files = line.split(" ")[1].rstrip()
            t_files = line.split(" ")[2].rstrip()
            enrol_files.append(e_files)
            test_files.append(t_files)

        enrol_files = list(np.unique(np.array(enrol_files)))
        test_files = list(np.unique(np.array(test_files)))

        # Prepare enrol csv
        logger.info("preparing enrol csv")
        enrol_csv = []
        for f in tqdm(enrol_files):
            wav = os.path.join(data_folder, f)

            # Reading the signal (to retrieve duration in seconds)
            signal, fs = torchaudio.load(wav)
            signal = signal.squeeze(0)
            audio_duration = signal.shape[0] / SAMPLERATE
            start_sample = 0
            stop_sample = signal.shape[0]
            wav_name = os.path.basename(wav).strip('.wav')
            id = wav_name
            [spk_id, sess_id, device_id, env_id, lang_id, style_id] = [wav_name[:4], wav_name[5], wav_name[6:9], wav_name[9], wav_name[10:12], wav_name[12]]

            csv_line = [
                id,
                audio_duration,
                wav,
                start_sample,
                stop_sample,
                spk_id,
            ]

            enrol_csv.append(csv_line)

        csv_output = csv_output_head + enrol_csv
        csv_file = os.path.join(save_folder, ENROL_CSV)

        # Writing the csv lines
        with open(csv_file, mode="w") as csv_f:
            csv_writer = csv.writer(
                csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            for line in csv_output:
                csv_writer.writerow(line)

        # Prepare test csv
        logger.info("preparing test csv")
        test_csv = []
        for f in tqdm(test_files):
            wav = os.path.join(data_folder, f)

            # Reading the signal (to retrieve duration in seconds)
            signal, fs = torchaudio.load(wav)
            signal = signal.squeeze(0)
            audio_duration = signal.shape[0] / SAMPLERATE
            start_sample = 0
            stop_sample = signal.shape[0]
            wav_name = os.path.basename(wav).strip('.wav')
            id = wav_name
            [spk_id, sess_id, device_id, env_id, lang_id, style_id] = [wav_name[:4], wav_name[5], wav_name[6:9], wav_name[9], wav_name[10:12], wav_name[12]]

            csv_line = [
                id,
                audio_duration,
                wav,
                start_sample,
                stop_sample,
                spk_id,
            ]

            test_csv.append(csv_line)

        csv_output = csv_output_head + test_csv
        csv_file = os.path.join(save_folder, TEST_CSV)

        # Writing the csv lines
        with open(csv_file, mode="w") as csv_f:
            csv_writer = csv.writer(
                csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            for line in csv_output:
                csv_writer.writerow(line)


def prepare_csv_enrol_test_hidden(data_folders, save_folder, verification_pairs_file):
    """
    Creates the csv file for test data (Hidden)

    Arguments
    ---------
    data_folder : str
        Path of the data folders
    save_folder : str
        The directory where to store the csv files.

    Returns
    -------
    None
    """

    # To implement after hidden data is released
    pass