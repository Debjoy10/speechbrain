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
import json
import torch
import logging
import torchaudio
import speechbrain as sb
from tqdm.contrib import tqdm
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.metric_stats import EER, minDCF
from speechbrain.utils.data_utils import download_file
from speechbrain.utils.distributed import run_on_main


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


def compute_embedding_loop(data_loader):
    """Computes the embeddings of all the waveforms specified in the
    dataloader.
    """
    embedding_dict = {}

    with torch.no_grad():
        for batch in tqdm(data_loader, dynamic_ncols=True):
            batch = batch.to(params["device"])
            seg_ids = batch.id
            wavs, lens = batch.sig
            # Restricting Size of Wav Tensor based on GPU size constraints
            wavs = wavs[:, :200000]

            found = False
            for seg_id in seg_ids:
                if seg_id not in embedding_dict:
                    found = True
            if not found:
                continue
            wavs, lens = wavs.to(params["device"]), lens.to(params["device"])
            emb = compute_embedding(wavs, lens).unsqueeze(1)
            for i, seg_id in enumerate(seg_ids):
                embedding_dict[seg_id] = emb[i].detach().clone()
    return embedding_dict


def get_verification_scores_test(veri_test):
    """Computes positive and negative scores given the test split."""
    scores = []
    save_file = os.path.join(params["output_folder"], "test_scores.txt" )
    s_file = open(save_file, "w")

    # Cosine similarity initialization
    similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    # creating cohort for score normalization
    if "score_norm" in params:
        train_cohort = torch.stack(list(train_dict.values()))

    for i, line in tqdm(enumerate(veri_test), total = len(veri_test)):

        # Reading verification file (enrol_file test_file label)
        enrol_id = line.split(" ")[0].rstrip().split(".")[0].strip()
        test_id = line.split(" ")[1].rstrip().split(".")[0].strip()
        enrol_idx = key2id['test_enrol'][enrol_id]
        test_idx = key2id['test_test'][test_id]

        # Compute Embeddings
        enrol_wavs = enrol_data[enrol_idx]['sig'][:200000].unsqueeze(0)
        enrol_lens = torch.tensor([1.])
        enrol_wavs, enrol_lens = enrol_wavs.to(params["device"]), enrol_lens.to(params["device"])
        enrol = compute_embedding(enrol_wavs, enrol_lens).unsqueeze(1).detach().clone()
        test_wavs = test_data[test_idx]['sig'][:200000].unsqueeze(0)
        test_lens = torch.tensor([1.])
        test_wavs, test_lens = test_wavs.to(params["device"]), test_lens.to(params["device"])
        test = compute_embedding(test_wavs, test_lens).unsqueeze(1).detach().clone()

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
        s_file.write("%s %s %f\n" % (enrol_id, test_id, score))
        scores.append(score)

    s_file.close()
    return scores


def get_verification_scores(veri_test):
    """Computes positive and negative scores given the verification split."""
    scores = []
    positive_scores = []
    negative_scores = []

    save_file = os.path.join(params["output_folder"], "scores.txt")
    s_file = open(save_file, "w")

    # Cosine similarity initialization
    similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    # creating cohort for score normalization
    if "score_norm" in params:
        train_cohort = torch.stack(list(train_dict.values()))

    for i, line in tqdm(enumerate(veri_test), total = len(veri_test)):

        # Reading verification file (enrol_file test_file label)
        lab_pair = int(line.split(" ")[0].rstrip().split(".")[0].strip())
        enrol_id = line.split(" ")[1].rstrip().split(".")[0].strip()
        test_id = line.split(" ")[2].rstrip().split(".")[0].strip()
        enrol_idx = key2id['enrol'][enrol_id]
        test_idx = key2id['test'][test_id]

        # Compute Embeddings
        enrol_wavs = enrol_data[enrol_idx]['sig'][:200000].unsqueeze(0)
        enrol_lens = torch.tensor([1.])
        enrol_wavs, enrol_lens = enrol_wavs.to(params["device"]), enrol_lens.to(params["device"])
        enrol = compute_embedding(enrol_wavs, enrol_lens).unsqueeze(1).detach().clone()
        test_wavs = test_data[test_idx]['sig'][:200000].unsqueeze(0)
        test_lens = torch.tensor([1.])
        test_wavs, test_lens = test_wavs.to(params["device"]), test_lens.to(params["device"])
        test = compute_embedding(test_wavs, test_lens).unsqueeze(1).detach().clone()

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

        if lab_pair == 1:
            positive_scores.append(score)
        else:
            negative_scores.append(score)

    s_file.close()
    return positive_scores, negative_scores

# Either normal verification eval or test eval - Specify
def dataio_prep(params):
    "Creates the dataloaders and their data processing pipelines."

    vox1_data_folder = params["vox1_data_folder"]
    vox2_data_folder = params["vox2_data_folder"]
    vox2022_data_folder = params["vox2022_data_folder"]

    # Train data (used for normalization)
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=params["train_data"], replacements={"data_root": vox2_data_folder},
    )
    train_data = train_data.filtered_sorted(
        sort_key="duration", select_n=params["n_train_snts"]
    )

    # Enrol data
    enrol_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=params["enrol_data"], replacements={"data_root": vox1_data_folder},
    )
    enrol_data = enrol_data.filtered_sorted(sort_key="duration")

    # Test data
    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=params["test_data"], replacements={"data_root": vox1_data_folder},
    )
    test_data = test_data.filtered_sorted(sort_key="duration")

    ############ TEST SPLIT
    # Enrol data
    test_enrol_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=params["test_enrol_data"], replacements={"data_root": vox2022_data_folder},
    )
    test_enrol_data = test_enrol_data.filtered_sorted(sort_key="duration")

    # Test data
    test_test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=params["test_test_data"], replacements={"data_root": vox2022_data_folder},
    )
    test_test_data = test_test_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, enrol_data, test_data, test_enrol_data, test_test_data]

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
        sig = sig.transpose(0, 1).squeeze(1)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # Set output
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig"])

    # Make dicts mapping ids back to indices
    key2id_Data = {}
    print("Collecting Data Indices")
    if os.path.isfile(os.path.join(params["output_folder"], "key2ids.json")):
        with open(os.path.join(params["output_folder"], "key2ids.json"), 'r') as fp:
            key2id_Data = json.load(fp)
    
    for split in ['train', 'enrol', 'test', 'test_enrol', 'test_test']:
        if split in key2id_Data:
            continue
        data = eval(split + '_data')
        try:
            _ = data[0]
            key2id_Data[split] = {data[i]['id']: i for i in tqdm(range(len(data)))}
        except:
            print("Cannot retrieve {} data".format(split))

    return train_data, enrol_data, test_data, test_enrol_data, test_test_data, key2id_Data


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

    test_veri_file_path = os.path.join(
        params["save_folder"], os.path.basename(params["test_verification_file"])
    )
    download_file(params["test_verification_file"], test_veri_file_path)

    from absp_voxceleb_prepare import prepare_voxceleb  # noqa E402

    # Create experiment directory
    sb.core.create_experiment_directory(
        experiment_directory=params["output_folder"],
        hyperparams_to_save=params_file,
        overrides=overrides,
    )

    # Prepare data from dev of Voxceleb1
    prepare_voxceleb(
        vox2022_data_folder=params["vox2022_data_folder"],
        vox2_data_folder=params["vox2_data_folder"],
        vox1_data_folder=params["vox1_data_folder"],
        save_folder=params["save_folder"],
        verification_pairs_file=veri_file_path,
        test_verification_pairs_file=test_veri_file_path,
        splits=["train", "dev", "test", "test2022"],
        split_ratio=[90, 10],
        seg_dur=3.0,
        source=params["voxceleb_source"]
        if "voxceleb_source" in params
        else None,
    )

    # here we create the datasets objects as well as tokenization and encoding
    train_data, enrol_data, test_data, enrol_data_2022, test_data_2022, key2id = dataio_prep(params)

    # We download the pretrained LM from HuggingFace (or elsewhere depending on
    # the path given in the YAML file). The tokenizer is loaded at the same time.
    run_on_main(params["pretrainer"].collect_files)
    params["pretrainer"].load_collected(params["device"])
    params["embedding_model"].eval()
    params["embedding_model"].to(params["device"])

    if params["test"]:
        # Reading standard verification split
        with open(test_veri_file_path) as f:
            veri_test = [line.rstrip() for line in f]
        scores = get_verification_scores_test(veri_test)
    else:
        # Compute the EER
        logger.info("Computing EER..")
        # Reading standard verification split
        with open(veri_file_path) as f:
            veri_test = [line.rstrip() for line in f]

        # Verification
        positive_scores, negative_scores = get_verification_scores(veri_test)
        eer, th = EER(torch.tensor(positive_scores), torch.tensor(negative_scores))
        logger.info("EER(%%)=%f", eer * 100)

        # Prevent CPU overflow
        min_dcf, th = minDCF(
            torch.tensor(positive_scores[:10000]), torch.tensor(negative_scores[:10000])
        )
        logger.info("minDCF=%f", min_dcf * 100)
