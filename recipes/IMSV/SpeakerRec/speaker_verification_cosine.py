#!/usr/bin/python3
"""Recipe for training a speaker verification system based on cosine distance.
The cosine distance is computed on the top of pre-trained embeddings.
The pre-trained model is automatically downloaded from the web if not specified.
This recipe is designed to work on a single GPU.

To run this recipe, run the following command:
    >  python speaker_verification_cosine.py hyperparams/verification_ecapa_tdnn.yaml

Authors
    * Hwidong Na 2020
    * Mirco Ravanelli 2020
"""
import os
import sys
import torch
import logging
import warnings
import random
import torchaudio
import speechbrain as sb
from tqdm.contrib import tqdm
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.metric_stats import EER, minDCF
from speechbrain.utils.data_utils import download_file
from speechbrain.utils.distributed import run_on_main
from tabulate import tabulate

# For reproducing random crops
random.seed(0)

# Compute embeddings from the waveforms
def compute_embedding(wavs, wav_lens):
    """Compute speaker embeddings.

    Arguments
    ---------
    wavs : Torch.Tensor
        Tensor containing the speech waveform (batch, time).
        Make sure the sample rate is fs=16000 Hz.
    wav_lens: Torch.Tensor
        Tensor containing the relative length for each sentence
        in the length (e.g., [0.8 0.6 1.0])
    """
    with torch.no_grad():
        feats = params["compute_features"](wavs)
        feats = params["mean_var_norm"](feats, wav_lens)
        embeddings = params["embedding_model"](feats, wav_lens)
    return embeddings.squeeze(1)


def compute_embedding_loop(data_loader, embedding_dict = {}):
    """Computes the embeddings of all the waveforms specified in the
    dataloader.
    """
    multivote = True    
    with torch.no_grad():
        for batch in tqdm(data_loader, dynamic_ncols=True):
            batch = batch.to(params["device"])
            seg_ids = batch.id
            wavs, lens = batch.sig
            # Restricting Size of Wav Tensor based on GPU size constraints to 20 sec
            
            # Do this iteratively for all crops
            if not multivote or int(lens[0]*wavs.shape[1]) < 320000:
                # First 20 sec
                wavs = wavs[:, :320000]

                found = False
                for seg_id in seg_ids:
                    if seg_id not in embedding_dict:
                        found = True
                if not found:
                    continue
                lens = torch.tensor([min(1, l*wavs.shape[1]/320000) for l in lens])
                wavs, lens = wavs.to(params["device"]), lens.to(params["device"])
                emb = compute_embedding(wavs, lens).unsqueeze(1)
                for i, seg_id in enumerate(seg_ids):
                    embedding_dict[seg_id] = emb[i].detach().cpu().clone()
            else:
                # Average over multiple segments
                segment_len = 320000
                n_segs = min(1 + int(lens[0]*wavs.shape[1]/segment_len), 20)
                start_times = random.sample(range(0, int(lens[0]*wavs.shape[1]) - segment_len), n_segs)

                # Check if a new audio signal is encountered
                found = False
                for seg_id in seg_ids:
                    if seg_id not in embedding_dict:
                        found = True
                if not found:
                    continue

                lens = torch.ones_like(lens)
                for k, start in enumerate(start_times):
                    wav = wavs[:, start:start+segment_len]
                    wav, lens = wav.to(params["device"]), lens.to(params["device"])
                    emb = compute_embedding(wav, lens).unsqueeze(1)
                    for i, seg_id in enumerate(seg_ids):
                        if k == 0:
                            embedding_dict[seg_id] = emb[i].detach().cpu().clone()
                        else:
                            embedding_dict[seg_id] = (embedding_dict[seg_id]*k + emb[i].detach().cpu().clone())/(k+1)
    return embedding_dict


def get_verification_scores(veri_test):
    """Computes positive and negative scores given the verification split."""
    scores = []
    positive_scores = {"ALL": []}
    negative_scores = {"ALL": []}
    positive_scores_equip = {}
    negative_scores_equip = {}

    save_file = os.path.join(params["output_folder"], "scores.txt")
    # File format changed from write to append to allow multiple test runs to write into the same scores file
    s_file = open(save_file, "w")

    # Cosine similarity initialization
    similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    # creating cohort for score normalization
    if "score_norm" in params:
        train_cohort = torch.stack(list(train_dict.values()))

    print("Computing Scores ...")
    for i, line in tqdm(enumerate(veri_test), total = len(veri_test)):

        # Reading verification file (enrol_file test_file label)
        lab_pair = int(line.split(" ")[0].rstrip().split(".")[0].strip())
        enrol_id = os.path.basename(line.split(" ")[1].rstrip()).split('.')[0]
        test_id = os.path.basename(line.split(" ")[2].rstrip()).split('.')[0]
        enrol = enrol_test_dict[enrol_id]
        test = enrol_test_dict[test_id]

        if "score_norm" in params:
            # Getting norm stats for enrol impostors
            enrol_rep = enrol.repeat(train_cohort.shape[0], 1, 1)
            score_e_c = similarity(enrol_rep, train_cohort)

            if "cohort_size" in params:
                score_e_c = torch.topk(
                    score_e_c, k=params["cohort_size"], dim=0
                )[0]

            mean_e_c = torch.mean(score_e_c, dim=0)
            std_e_c = torch.std(score_e_c, dim=0)

            # Getting norm stats for test impostors
            test_rep = test.repeat(train_cohort.shape[0], 1, 1)
            score_t_c = similarity(test_rep, train_cohort)

            if "cohort_size" in params:
                score_t_c = torch.topk(
                    score_t_c, k=params["cohort_size"], dim=0
                )[0]

            mean_t_c = torch.mean(score_t_c, dim=0)
            std_t_c = torch.std(score_t_c, dim=0)

        # Compute the score for the given sentence
        score = similarity(enrol, test)[0]

        # Perform score normalization
        if "score_norm" in params:
            if params["score_norm"] == "z-norm":
                score = (score - mean_e_c) / std_e_c
            elif params["score_norm"] == "t-norm":
                score = (score - mean_t_c) / std_t_c
            elif params["score_norm"] == "s-norm":
                score_e = (score - mean_e_c) / std_e_c
                score_t = (score - mean_t_c) / std_t_c
                score = 0.5 * (score_e + score_t)

        # write score file
        s_file.write("%s %s %i %f\n" % (enrol_id, test_id, lab_pair, score))
        scores.append(score)

        # Which Language
        lang = str(test_id[-3:-1])
        equip = str(test_id[-7:-4])
        
        if lab_pair == 1:
            positive_scores["ALL"].append(score)
            if lang not in positive_scores:
                positive_scores[lang] = []
            positive_scores[lang].append(score)
            if equip not in positive_scores_equip:
                positive_scores_equip[equip] = []
            positive_scores_equip[equip].append(score)
        else:
            negative_scores["ALL"].append(score)
            if lang not in negative_scores:
                negative_scores[lang] = []
            negative_scores[lang].append(score)
            if equip not in negative_scores_equip:
                negative_scores_equip[equip] = []
            negative_scores_equip[equip].append(score)

    s_file.close()
    logger.info("Scores written to {}".format(save_file))
    return positive_scores, negative_scores, positive_scores_equip, negative_scores_equip


def dataio_prep(params):
    "Creates the dataloaders and their data processing pipelines."

    data_folder = params["data_folder"]

    # Train data (used for normalization)
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=params["train_data"], replacements={"data_root": data_folder},
    )
    train_data = train_data.filtered_sorted(
        sort_key="duration", select_n=params["n_train_snts"]
    )

    # Enrol data
    enrol_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=params["enrol_data"], replacements={"data_root": data_folder},
    )
    enrol_data = enrol_data.filtered_sorted(sort_key="duration")

    # Test data
    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=params["test_data"], replacements={"data_root": data_folder},
    )
    test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, enrol_data, test_data]

    # Define audio pipeline
    @sb.utils.data_pipeline.takes("wav", "start", "stop")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, start, stop):
        start = int(start)
        stop = int(stop)
        num_frames = stop - start
        sig, fs = torchaudio.load(
            wav, num_frames=num_frames, frame_offset=start
        )
        # Add resampling HERE
        if fs != 16000:
            warnings.warn("Data with different sampling frequency present, might reduce speed")
            sig = torchaudio.functional.resample(sig, orig_freq=fs, new_freq=16000)
        sig = sig.transpose(0, 1).squeeze(1)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # Set output
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig"])

    # Create dataloaders
    train_dataloader = sb.dataio.dataloader.make_dataloader(
        train_data, **params["train_dataloader_opts"]
    )
    enrol_dataloader = sb.dataio.dataloader.make_dataloader(
        enrol_data, **params["enrol_dataloader_opts"]
    )
    test_dataloader = sb.dataio.dataloader.make_dataloader(
        test_data, **params["test_dataloader_opts"]
    )

    return train_dataloader, enrol_dataloader, test_dataloader


if __name__ == "__main__":
    # Logger setup
    logger = logging.getLogger(__name__)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(current_dir))

    # Load hyperparameters file with command-line overrides
    params_file, run_opts, overrides = sb.core.parse_arguments(sys.argv[1:])
    with open(params_file) as fin:
        params = load_hyperpyyaml(fin, overrides)

    # Download verification list (to exlude verification sentences from train)
    veri_file_path = os.path.join(
        params["save_folder"], os.path.basename(params["verification_file"])
    )
    download_file(params["verification_file"], veri_file_path)

    from imsv_prepare import prepare_imsv  # noqa

    # Create experiment directory
    sb.core.create_experiment_directory(
        experiment_directory=params["output_folder"],
        hyperparams_to_save=params_file,
        overrides=overrides,
    )

    # Prepare data from dev of Voxceleb1
    prepare_imsv(
        data_folder=params["data_folder"],
        save_folder=params["save_folder"],
        verification_pairs_file=veri_file_path,
        test_pairs_file=None,
        hidden_test_pairs_file=None,
        splits=["train", "dev", "enr"],
        split_ratio=[90, 10],
        seg_dur=2.0,
    )

    # here we create the datasets objects as well as tokenization and encoding
    train_dataloader, enrol_dataloader, test_dataloader = dataio_prep(params)

    # We download the pretrained LM from HuggingFace (or elsewhere depending on
    # the path given in the YAML file). The tokenizer is loaded at the same time.
    run_on_main(params["pretrainer"].collect_files)
    params["pretrainer"].load_collected(params["device"])
    params["embedding_model"].eval()
    params["embedding_model"].to(params["device"])

    # Computing  enrollment and test embeddings
    logger.info("Computing enroll/test embeddings...")

    # First run
    # Shared enrol-test dict
    enrol_test_dict = {}
    enrol_test_dict = compute_embedding_loop(enrol_dataloader, enrol_test_dict)
    enrol_test_dict = compute_embedding_loop(test_dataloader, enrol_test_dict)

    if "score_norm" in params:
        train_dict = compute_embedding_loop(train_dataloader)

    # Reading standard verification split
    with open(veri_file_path) as f:
        veri_test = [line.rstrip() for line in f]

    positive_scores_lang, negative_scores_lang, positive_scores_equip, negative_scores_equip = get_verification_scores(veri_test)
    del enrol_test_dict

    if not params["test"]:
        # Compute the EER
        for (positive_scores, negative_scores, label) in [(positive_scores_lang, negative_scores_lang, "Language"), 
                                                  (positive_scores_equip, negative_scores_equip, "Sensor")]:
            score_Dict = {}
            for key in positive_scores.keys():
                # logger.info("Computing EER for {} language ..".format(key))
                eer, th = EER(torch.tensor(positive_scores[key]), torch.tensor(negative_scores[key]))
                # logger.info("EER(%%)=%f", eer * 100)

                min_dcf, th = minDCF(
                    torch.tensor(positive_scores[key]), torch.tensor(negative_scores[key])
                )
                # logger.info("minDCF=%f", min_dcf * 100)
                score_Dict[key] = {}
                score_Dict[key]["EER"] = eer * 100
                score_Dict[key]["minDCF"] = min_dcf * 100

            eer_row = ["   EER"] + [v["EER"] for k, v in score_Dict.items()]
            mindcf_row = ["minDCF"] + [v["minDCF"] for k, v in score_Dict.items()]
            print(tabulate([eer_row, mindcf_row], headers=[label]+list(score_Dict.keys())))