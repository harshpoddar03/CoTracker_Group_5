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
from complete_validation_utils import forward_window
from torch.utils.tensorboard import SummaryWriter
from cotracker.predictor2 import CoTrackerPredictor  # Import the class
from cotracker.models.core.cotracker.cotracker import CoTracker2
import itertools
import normal_cotracker
from cotracker.datasets.utils import collate_fn, collate_fn_train, dataclass_to_cuda_
from modified_loss import sequence_loss, balanced_ce_loss
curr_collate_fn = collate_fn
def validate(model,model_name, train_dataloader, num_steps, dataset_name,step_num, logs_dir, avg_values_at_100, model_type):

    writer = SummaryWriter(logs_dir)
    step = 0
    data_iter = iter(train_dataloader)
    # print(len(next(data_iter)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    losses_at_100 = []
    davg_at_100 = []
    losses_at_1000 = []
    davg_at_1000 = []
    data = f"{dataset_name}_{step_num}"
    model.cuda()
    if model_name == "kp_embed" and (dataset_name == "KubricsTrain" or dataset_name == "KubricsVal" or dataset_name == "TapVidDavis"):
        model.skip = True
    elif model_name == "kp_embed" and (dataset_name == "PoseTrackTrain" or dataset_name == "PoseTrackVal"):
        model.skip = False
    with torch.no_grad():
        while step < num_steps:
            model.eval()
            start_time = time.time()
            try:
                if dataset_name == "PoseTrackTrain" or dataset_name == "PoseTrackVal" or dataset_name == "PoseTrackTrain_Pyscene" or dataset_name =="PoseTrackVal_Pyscene":
                    video, queries, trajs_g, vis_g, valids = next(data_iter)
                elif dataset_name == "KubricsTrain" or dataset_name == "KubricsVal":
                    batch, gotit = next(data_iter)
                elif dataset_name == "TapVidDavis":
                    sample = next(data_iter)
                else:
                    print("No exisiting dataset matched")
                    break

            except StopIteration:
                data_iter = iter(train_dataloader)

                if dataset_name == "PoseTrackTrain" or dataset_name == "PoseTrackVal" or dataset_name == "PoseTrackTrain_Pyscene" or dataset_name =="PoseTrackVal_Pyscene":
                    video, queries, trajs_g, vis_g, valids = next(data_iter)
                elif dataset_name == "KubricsTrain" or dataset_name == "KubricsVal":
                    batch, gotit = next(data_iter)
                elif dataset_name == "TapVidDavis":
                    sample = next(data_iter)
                else:
                    print("No exisiting dataset matched")
                    break
            except OSError as e:
                print(f"Error occurred during data loading: {e}")
                # Optionally handle the error, e.g., skip the current batch and continue
                continue
            except Exception as e:
                print(f"Unexpected error: {e}")
                # Optionally handle other exceptions if needed
                continue
            if dataset_name == "KubricsTrain" or dataset_name == "KubricsVal":
                if not all(gotit):
                    print("batch is None")
                    continue

                video = batch.video
                trajs_g = batch.trajectory
                vis_g = batch.visibility
                valids = batch.valid
            elif dataset_name == "TapVidDavis":
                video = sample.video.clone()
                vis_g = sample.visibility.clone()
                trajs_g = sample.trajectory.clone()
                valids = None
            if torch.cuda.is_available():
                video = video.cuda()
                trajs_g = trajs_g.cuda()
                vis_g = vis_g.cuda()
                if valids is not None:
                    valids = valids.cuda()
            # pred_tracks, loss, step_d_avg = forward_window(model, video,trajs_g, vis_g, valids, sequence_length = 15, window_length = 8, iters = 4)
            T = video.shape[1]
            loss, step_d_avg = forward_window(model, video,trajs_g, vis_g, valids, sequence_length = T, window_length = 8, iters = 4)
            if torch.isnan(loss):
                print(f"Loss is NaN, Step {step}")
                continue
            losses_at_100.append(loss.item())
            writer.add_scalar(f'Step Loss Validation {data}', loss.item(), step)
            writer.add_scalar(f'd_avg Step Metric Validation {data}', step_d_avg, step)
            if not (np.isnan(step_d_avg)):
                davg_at_100.append(step_d_avg)
            end_time = time.time()
            step_time = end_time - start_time
            if step%50 == 0:
                print(f"Optimized {data} {model_type}: Step {step}, Loss: {loss.item():.4f}, d_avg: {step_d_avg:.4f}, step_time : {step_time:.4f}")
            if dataset_name == "TapVidDavis":
                print(f"Optimized {data} {model_type}: Step {step}, Loss: {loss.item():.4f}, d_avg: {step_d_avg:.4f}, step_time : {step_time:.4f}")
                with open(avg_values_at_100, "a") as file:
                        file.write(f"Optimized: Steps {step+1}, Loss: {loss.item()}, d_avg: {step_d_avg}\n")
            if dataset_name != "TapVidDavis":
                if (step+1) % 100 == 0:
                    losses_at_1000 += losses_at_100
                    davg_at_1000 += davg_at_100
                    avgloss100 = sum(losses_at_100) / len(losses_at_100)
                    losses_at_100 = []
                    avgdavg100 = sum(davg_at_100) / len(davg_at_100)
                    davg_at_100 = []
                    print(f"Optimized {data} {model_type} (At 100): Steps {step+1}, Loss: {avgloss100}, d_avg: {avgdavg100}")
                    with open(avg_values_at_100, "a") as file:
                        file.write(f"Optimized (At 100): Steps {step+1}, Loss: {avgloss100}, d_avg: {avgdavg100}\n")
                    writer.add_scalar(f'Avg_loss_at_every_100_Validation_{data}', avgloss100, step+1)
                    writer.add_scalar(f'Avg_d_avg_at_every_100_Validation_{data}', avgdavg100, step+1)
                    # break

                
            step += 1
    if dataset_name != "TapVidDavis":
        avgloss1000 = sum(losses_at_1000) / len(losses_at_1000)
        avgdavg1000 = sum(davg_at_1000) / len(davg_at_1000)
        final_avg = avgdavg1000
        final_loss = avgloss1000
        print(f"Finally, Optimized {data} {model_type} (At 1000): Steps {step+1}, Loss: {avgloss1000}, d_avg: {avgdavg1000}")
        with open(avg_values_at_100, "a") as file:
            file.write(f"Finally, Optimized (At 1000): Steps {step+1}, Loss: {avgloss1000}, d_avg: {avgdavg1000}\n")
    else:
        avgloss100 = sum(losses_at_100) / len(losses_at_100)
        losses_at_100 = []
        avgdavg100 = sum(davg_at_100) / len(davg_at_100)
        davg_at_100 = []
        final_avg = avgdavg100
        final_loss = avgloss100
        print(f"Optimized (At 30) {data} {model_type} : Steps {step+1}, Loss: {avgloss100}, d_avg: {avgdavg100}")
        with open(avg_values_at_100, "a") as file:
            file.write(f"Finally Optimized (At 30): Steps {step+1}, Loss: {avgloss100}, d_avg: {avgdavg100}\n")

    writer.close()
    return final_avg, final_loss


