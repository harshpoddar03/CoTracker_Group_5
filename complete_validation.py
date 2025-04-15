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
from val import validate
from complete_validation_utils import generate_dirs, load_weights, generate_model, generate_model, generate_dataloader
if __name__ == "__main__":
    # model_names = ["kubrics0_posetrack100_absent",
    #                "kubrics0_posetrack100_present",
    #                "kubrics50_posetrack50_present",
    #                "kubrics75_posetrack25_present",
    #                "kubrics0_posetrack100_absent_newdloader"]
    model_names = ["kubrics0_posetrack100_absent_newdloader_scratch"]
    dataset_names = ["PoseTrackTrain","PoseTrackVal", "PoseTrackTrain_Pyscene","PoseTrackVal_Pyscene","KubricsTrain", "KubricsVal","TapVidDavis"]
    # dataset_names = ["PoseTrackTrain_Pyscene"]
    finale_path = "/home/anudeep/final_values"
    step_nums = [30000, 40000, 60000]
    num_steps_val = 1000
    num_steps_to_val = 1000
    for model_name in model_names:
        logs_dir, ckpt_dir = generate_dirs(model_name)
        # print(ckpt_dir)
        if "present" in model_name:
            model = generate_model(model_type = "kp_embed")
            z = "kp_embed"
        else:
            model = generate_model(model_type = "normal")
            z = "normal"
        for dataset in dataset_names:
            dataloader = generate_dataloader(dataset,num_steps_val)
            if(dataset == "TapVidDavis"):
                num_steps_to_val = len(dataloader)
            else:
                num_steps_to_val = 1000
            # if model_name == "kubrics0_posetrack100_present" and (dataset == "PoseTrackTrain" or dataset == "PoseTrackVal"):
            #     continue
            # print(dataset)
            for step in step_nums:
                # if model_name == "kubrics0_posetrack100_absent" and (dataset == "KubricsTrain") and step == 30000:
                #     continue
                if ckpt_dir is None:
                    with open("/home/anudeep/ckpt/cotracker2.pth", "rb") as f:
                        state_dict = torch.load(f, map_location="cpu")
                        if "model" in state_dict:
                            state_dict = state_dict["model"]
                        model.load_state_dict(state_dict)
                else:
                    ckpt_path = load_weights(ckpt_dir, step)
                    checkpoint = torch.load(ckpt_path)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    start_step = checkpoint['step']
                    print(f"Loaded model from {ckpt_path}, starting from step {start_step}")
                    data = f"{dataset}_{step}"
                    new_path = f'{"/".join(logs_dir.split("/")[:-1])}/values_at_100_{data}'
                final_avg, final_loss = validate(model,z,dataloader, num_steps_to_val, dataset, step, logs_dir, new_path, model_name)
                print(f"Finally optimized, {data}, on model {model_name}, got davg as {final_avg}, got loss as {final_loss}")
                with open(finale_path, "a") as file:
                    file.write(f"Finally optimized, {data}, on model {model_name}, got davg as {final_avg}, got loss as {final_loss}\n")