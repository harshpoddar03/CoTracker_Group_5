import os
from base64 import b64encode
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from IPython.display import HTML
from utils import set_random_seeds, show_video, loss_fn, reduce_masked_mean, evaluate_trajectories
from dataloader2 import PoseTrackDataset
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import random
import torch
import new_datasets
import time
import new_tapvid
from cotracker.datasets.utils import collate_fn

from torch.utils.tensorboard import SummaryWriter
from cotracker.predictor2 import CoTrackerPredictor  # Import the class
from cotracker.models.core.cotracker.cotracker import CoTracker2
import itertools
import normal_cotracker
from cotracker.datasets.utils import collate_fn, collate_fn_train, dataclass_to_cuda_
from modified_loss import sequence_loss, balanced_ce_loss
curr_collate_fn = collate_fn

def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def generate_dirs(model_name):
    if model_name == "baseline":
        logs_dir = '/home/anudeep/kubric_dataset/metrics_normal/posetrack_logs'
        return (logs_dir, None)
    elif model_name == "kubrics100_posetrack0_absent":
        logs_dir = "/home/anudeep/kubric_dataset/metrics_kubrics/kubrics_logs"
        ckpt_dir = "/home/anudeep/kubric_dataset/metrics_kubrics/kubrics_checkpoints"
        return logs_dir, ckpt_dir
    elif model_name == "kubrics0_posetrack100_absent":
        logs_dir = "/home/anudeep/kubric_dataset/metrics_wokp_posetrack/posetrack_logs"
        ckpt_dir = "/home/anudeep/kubric_dataset/metrics_wokp_posetrack/posetrack_checkpoints"
        return logs_dir, ckpt_dir
    elif model_name == "kubrics0_posetrack100_present":
        logs_dir = "/home/anudeep/kubric_dataset/metrics_posetrack/posetrack_logs"
        ckpt_dir = "/home/anudeep/kubric_dataset/metrics_posetrack/posetrack_checkpoints"
        return logs_dir, ckpt_dir
    elif model_name == "kubrics50_posetrack50_present":
        logs_dir = "/home/anudeep/kubric_dataset/metrics_p+k/p+k_logs"
        ckpt_dir = "/home/anudeep/kubric_dataset/metrics_p+k/p+k_checkpoints"
        return logs_dir, ckpt_dir
    elif model_name == "kubrics75_posetrack25_present":
        logs_dir = "/home/anudeep/kubric_dataset/metrics_p+k75/p+k_logs"
        ckpt_dir = "/home/anudeep/kubric_dataset/metrics_p+k75/p+k_checkpoints"
        return logs_dir, ckpt_dir
    elif model_name == "kubrics0_posetrack100_absent_newdloader":
        logs_dir = "/home/anudeep/kubric_dataset/metrics_wokp_posetrack_newdataloader/posetrack_logs"
        ckpt_dir = "/home/anudeep/kubric_dataset/metrics_wokp_posetrack_newdataloader/posetrack_checkpoints"
        return logs_dir, ckpt_dir
    elif model_name == "kubrics0_posetrack100_absent_newdloader_scratch":
        logs_dir = "/home/anudeep/kubric_dataset/metrics_wokp_posetrack_newdataloader_scratch/posetrack_logs"
        ckpt_dir = "/home/anudeep/kubric_dataset/metrics_wokp_posetrack_newdataloader_scratch/posetrack_checkpoints"
        return logs_dir, ckpt_dir
    return None,None

def load_weights(ckpt_dir, step_num = None):
    checkpoints = [f for f in os.listdir(ckpt_dir) if f.endswith('.pth')]
    if(step_num == None):
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[2].split('.')[0]))
        checkpoint_path = os.path.join(ckpt_dir, latest_checkpoint)
    else:
        model_name = f"model_step_{step_num-1}.pth"
        checkpoint_path = os.path.join(ckpt_dir, model_name)
    
    return checkpoint_path

def generate_model(model_type):
    if(model_type == "normal"):
        model = normal_cotracker.CoTracker2(stride=4, window_len=8, add_space_attn=True)
        return model
    elif(model_type == "kp_embed"):
        model = CoTracker2(stride=4, window_len=8, add_space_attn=True, skip = False)
        return model
    return None

def generate_dataloader(dataset_name, num_samples = None):
    if(dataset_name == "PoseTrackTrain" or dataset_name == "PoseTrackVal" or dataset_name =="PoseTrackTrain_Pyscene" or dataset_name =="PoseTrackVal_Pyscene"):
        if dataset_name == "PoseTrackTrain":
            file = "train"
            json_file = "train_15"
        elif dataset_name == "PoseTrackVal":
            file = "val"
            json_file = "val_15"
        elif dataset_name == "PoseTrackTrain_Pyscene":
            file = "train"
            json_file = "train_15_pyscene"
        else:
            file = "val"
            json_file = "val_15_pyscene"
            # print("heyyyyy")
        set_random_seeds()
        train_folder = f'/home/anudeep/dataset/PoseTrack21/data/images/{file}'
        train_json_folder = f'/home/anudeep/dataset/PoseTrack21/data/PoseTrack21/posetrack_data/{file}'
        train_start_frames_folder = f'/home/anudeep/dataset/PoseTrack21/data/{json_file}.json'
        train_dataset = PoseTrackDataset(train_folder, train_json_folder,train_start_frames_folder, 15, (384,512))
        indices = torch.randperm(len(train_dataset)).tolist()
        subset_indices = indices[:num_samples]
        subset = Subset(train_dataset, subset_indices)
        posetrack_dataloader = DataLoader(subset, batch_size=1, num_workers = 4)
        return posetrack_dataloader
    elif(dataset_name == "KubricsTrain" or dataset_name ==  "KubricsVal"):
        if dataset_name == "KubricsTrain":
            file = "train"
        else:
            file = "val"
        g = torch.Generator()
        g.manual_seed(0)
        train_dataset = new_datasets.KubricMovifDataset(
            # data_root=os.path.join(args.dataset_root, "kubric", "kubric_movi_f_tracks"),
            data_root = f"/home/anudeep/kubric_dataset/kubric_movi_f_{file}",
            crop_size=[384, 512],
            seq_len=15,
            traj_per_sample=100,
            sample_vis_1st_frame=False,
            use_augs=True,
        )
        indices = torch.randperm(len(train_dataset)).tolist()
        subset_indices = indices[:num_samples]
        subset = Subset(train_dataset, subset_indices)
        kubrics_dataloader = DataLoader(
                    subset,
                    batch_size=1,
                    shuffle=True,
                    num_workers=4,
                    worker_init_fn=seed_worker,
                    generator=g,
                    pin_memory=True,
                    collate_fn=collate_fn_train,
                    drop_last=True,
                )
        # print("hello")  
        return kubrics_dataloader
    elif(dataset_name == "TapVidDavis"):
        set_random_seeds()
        test_dataset = new_tapvid.TapVidDataset(
            data_root="/home/anudeep/kubric_dataset/tapvid_davis/tapvid_davis.pkl",
            # resize_to_256=False,
            crop_size = (384,512)
        )
        tapvid_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=4,
                collate_fn=curr_collate_fn,
            )
        return tapvid_dataloader
    return None

def forward_window(model, video,trajs_g, vis_g, valids, sequence_length, window_length, iters):
    B, T, C, H, W = video.shape
    assert C == 3
    B, T, N, D = trajs_g.shape
    device = video.device
    __, first_positive_inds = torch.max(vis_g, dim=1)
    N_rand = N // 4
    nonzero_inds = [[torch.nonzero(vis_g[b, :, i]) for i in range(N)] for b in range(B)]
    # print(f"Non zero indices: {nonzero_inds}")
    # print(f"Initially : {first_positive_inds}")
    for b in range(B):
        rand_vis_inds = torch.cat(
            [
                # Check if nonzero_row is empty
                (nonzero_row[torch.randint(len(nonzero_row), size=(1,))]
                if len(nonzero_row) > 0
                else torch.tensor([[0]], dtype=torch.int, device = device))
                for nonzero_row in nonzero_inds[b]
            ],
            dim=1
        )
        # print(f"Random: {rand_vis_inds}")
        first_positive_inds[b] = torch.cat(
            [rand_vis_inds[:, :N_rand], first_positive_inds[b : b + 1, N_rand:]], dim=1
        )
        # print(f"Secondly : {first_positive_inds}")
    
    gather = torch.gather(trajs_g, 1, first_positive_inds[:, :, None, None].repeat(1, 1, N, D))
    xys = torch.diagonal(gather, dim1=1, dim2=2).permute(0, 2, 1)

    queries = torch.cat([first_positive_inds[:, :, None], xys[:, :, :2]], dim=2)

    predictions, visibility, train_data = model(
        video=video, queries=queries, iters=iters, is_train=True
    )
    coord_predictions, vis_predictions, valid_mask = train_data

    vis_gts = []
    traj_gts = []
    valids_gts = []

    S = window_length
    if valids is not None:
        for ind in range(0, sequence_length - S // 2, S // 2):
            # print(f"ind : {ind}")
            vis_gts.append(vis_g[:, ind : ind + S])
            traj_gts.append(trajs_g[:, ind : ind + S])
            valids_gts.append(valids[:, ind : ind + S] * valid_mask[:, ind : ind + S])
        # metrics = evaluate_trajectories(trajs_g, predictions.detach(), valids * vis_g, 512, 384)
        new_valids = valids * valid_mask
        metrics = evaluate_trajectories(trajs_g, predictions.detach(), new_valids, 512, 384)

    else:
        for ind in range(0, sequence_length - S // 2, S // 2):
            # print(f"ind : {ind}")
            vis_gts.append(vis_g[:, ind : ind + S])
            traj_gts.append(trajs_g[:, ind : ind + S])
            valids_gts.append(valid_mask[:, ind : ind + S])
        new_valids = valid_mask * vis_g
        metrics = evaluate_trajectories(trajs_g, predictions.detach(), valid_mask * vis_g, 512, 384)

    seq_loss = sequence_loss(coord_predictions, traj_gts, vis_gts, valids_gts, 0.8)
    
    step_d_avg = metrics['d_avg']

    return seq_loss.mean(), step_d_avg