import os
from base64 import b64encode
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from IPython.display import HTML
from utils import set_random_seeds, show_video, loss_fn, reduce_masked_mean, evaluate_trajectories,evaluate_trajectories_1scene
from new_posetrack_dataloader import PoseTrackDataset
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import torch
from dataloader2 import PoseTrackDataset
import time
from torch.utils.tensorboard import SummaryWriter
from cotracker.predictor2 import CoTrackerPredictor  # Import the class
from cotracker.models.core.cotracker.cotracker import CoTracker2
import itertools
import normal_cotracker
from modified_loss import sequence_loss, balanced_ce_loss
# model = CoTracker2(stride=4, window_len=8, add_space_attn=True)
model = normal_cotracker.CoTracker2(stride=4, window_len=8, add_space_attn=True, model_resolution=(768,576))

# with open("/home/anudeep/ckpt/cotracker2.pth", "rb") as f:
#   state_dict = torch.load(f, map_location="cpu")
#   if "model" in state_dict:
#       state_dict = state_dict["model"]
# # # model.load_state_dict(state_dict, strict = False)
# model.load_state_dict(state_dict)
# for name, param in model.named_parameters():
#     print(f"{name}: requires_grad={param.requires_grad}")

set_random_seeds()

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

train_folder = '/home/anudeep/PoseTrack21/data/images/train'
train_json_folder = '/home/anudeep/PoseTrack21/data/PoseTrack21/posetrack_data/train'
train_start_frames_folder = '/home/anudeep/PoseTrack21/data/train_15_pyscene.json'
train_dataset = PoseTrackDataset(train_folder, train_json_folder,train_start_frames_folder, 15, (384,512))
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers = 3)


def forward_window(model, video,trajs_g, vis_g, valids, sequence_length, window_length, iters):
    B, T, C, H, W = video.shape
    assert C == 3
    B, T, N, D = trajs_g.shape
    device = video.device
    __, first_positive_inds = torch.max(vis_g, dim=1)
    N_rand = N // 4
    nonzero_inds = [[torch.nonzero(vis_g[b, :, i]) for i in range(N)] for b in range(B)]

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
        first_positive_inds[b] = torch.cat(
            [rand_vis_inds[:, :N_rand], first_positive_inds[b : b + 1, N_rand:]], dim=1
        )
    
    gather = torch.gather(trajs_g, 1, first_positive_inds[:, :, None, None].repeat(1, 1, N, D))
    xys = torch.diagonal(gather, dim1=1, dim2=2).permute(0, 2, 1)

    queries = torch.cat([first_positive_inds[:, :, None], xys[:, :, :2]], dim=2)

    predictions, visibility, train_data = model(
        video=video, queries=queries, iters=iters, is_train=True
    )
    exit(1)
    coord_predictions, vis_predictions, valid_mask = train_data

    vis_gts = []
    traj_gts = []
    valids_gts = []
    new_valids = valid_mask*valids
    S = window_length
    for ind in range(0, sequence_length - S // 2, S // 2):
        # print(f"ind : {ind}")
        vis_gts.append(vis_g[:, ind : ind + S])
        traj_gts.append(trajs_g[:, ind : ind + S])
        valids_gts.append(valids[:, ind : ind + S] * valid_mask[:, ind : ind + S])

    seq_loss = sequence_loss(coord_predictions, traj_gts, vis_gts, valids_gts, 0.8)
    metrics = evaluate_trajectories_1scene(trajs_g, predictions.detach(), new_valids, 576, 768)
    # metrics = evaluate_trajectories(trajs_g, predictions.detach(), valids, 512, 384)
    step_d_avg = metrics['d_avg']
    return predictions.detach(),seq_loss.mean(),step_d_avg


EPS = 1e-6

def print_metrics(metrics):
  for key, value in metrics.items():
    print(f"{key}: {value}")

def save_metrics(metrics, save_dir, epoch):
  file_path = os.path.join(save_dir, 'metrics.txt')
  with open(file_path, 'a') as file:
    file.write(f'Epoch: {epoch}\n')
    # Iterate through the dictionary items
    for key, value in metrics.items():
        # Write each key-value pair on a new line
        file.write(f'{key}: {value}\n')

def make_video(pred_tracks, pred_visibility, video, vis_writer, epoch,batch_idx, idx):
  video = video[0][idx:]
  pred_tracks = pred_tracks[0][idx:]
  pred_visibility = pred_visibility[0][idx:]
  vis = Visualizer(
      linewidth=1,
      mode='cool',
      # save_dir=save_path
      # tracks_leave_trace=-1
  )
  vis.visualize(
      writer = vis_writer,
      video=video[None],
      tracks=pred_tracks[None],
      visibility=pred_visibility[None],
      step = epoch,
      filename=f'video_{batch_idx}')
import pickle

def train(model, optimizer,train_dataloader, load_model=False, new_lr=None, new_momentum=None, num_steps=9000000):
    set_random_seeds()

    logs_dir = '/home/anudeep/kubric_dataset/metrics_cotracker_scratch_768,576/posetrack_logs'
    ckpt_dir = '/home/anudeep/kubric_dataset/metrics_cotracker_scratch_768,576/posetrack_checkpoints'
    avg_values_at_100 = '/home/anudeep/kubric_dataset/metrics_cotracker_scratch_768,576/values_at_100'


    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok = True)

    writer = SummaryWriter(logs_dir)

    loss_threshold = 0.01
    start_step = 0
    data_iter = iter(train_dataloader)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if load_model:
        checkpoints = [f for f in os.listdir(ckpt_dir) if f.endswith('.pth')]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[2].split('.')[0]))
            checkpoint_path = os.path.join(ckpt_dir, latest_checkpoint)
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_step = checkpoint['step']
            print(f"Loaded model from {checkpoint_path}, starting from step {start_step}")

            if new_lr is not None:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                print(f"Learning rate updated to {new_lr}")

            if new_momentum is not None:
                if 'betas' in optimizer.param_groups[0]:  # For Adam/AdamW optimizers
                    for param_group in optimizer.param_groups:
                        param_group['betas'] = (new_momentum, param_group['betas'][1])
                    print(f"Momentum (beta1) updated to {new_momentum}")
                elif 'momentum' in optimizer.param_groups[0]:  # For SGD optimizers
                    for param_group in optimizer.param_groups:
                        param_group['momentum'] = new_momentum
                    print(f"Momentum updated to {new_momentum}") 

    step = start_step+1
    
    losses_at_100 = []
    davg_at_100 = []
    while step < num_steps:
        model.train()
        start_time = time.time()

        try:
            video, queries, trajs_g, vis_g, valids,gt_heatmaps = next(data_iter)
            # valids = vis_g
        except StopIteration:
            data_iter = iter(train_dataloader)
            video, queries, trajs_g, vis_g, valids,gt_heatmaps = next(data_iter)
            # valids = vis_g
        except OSError as e:
            print(f"Error occurred during data loading: {e}")
            # Optionally handle the error, e.g., skip the current batch and continue
            continue
        except Exception as e:
            print(f"Unexpected error: {e}")
            # Optionally handle other exceptions if needed
            continue
        if torch.cuda.is_available():
            video = video.cuda()
            trajs_g = trajs_g.cuda()
            vis_g = vis_g.cuda()
            valids = valids.cuda()
        # print(video.shape)
        # print(trajs_g.shape)
        # print(gt_heatmaps.shape)
        # exit(1)
       # pred_tracks, pred_visibility, _ = model(video, queries=queries)
        #loss = loss_fn(trajs_e, pred_tracks, valids)
        # print("hey")
        # print(f"Trajs : {trajs_e.shape}, Preds : {pred_tracks.shape}, Visi : {visibility.shape}, Valids : {valids.shape}")
        _, loss,step_d_avg = forward_window(model, video,trajs_g, vis_g, valids, sequence_length = 15, window_length = 8, iters = 4)
        if torch.isnan(loss):
            print(f"Loss is NaN, Step {step}")
            continue
        losses_at_100.append(loss.item())
        writer.add_scalar('Step Loss', loss.item(), step)
        writer.add_scalar('d_avg Step Metric', step_d_avg, step)
        if not (np.isnan(step_d_avg)):
            davg_at_100.append(step_d_avg)
        # davg_at_100.append(step_d_avg)
        if torch.isnan(loss):
            print(f"Loss is NaN, Step {step}")
            break

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        end_time = time.time()
        step_time = end_time - start_time
        # if step%50 == 0 or step ==1:
        print(f"Optimized: Step {step}, Loss: {loss.item():.4f}, d_avg: {step_d_avg:.4f}, step_time : {step_time:.4f}")

        if (step+1) % 100 == 0:
            avgloss100 = sum(losses_at_100) / len(losses_at_100)
            losses_at_100 = []
            avgdavg100 = sum(davg_at_100) / len(davg_at_100)
            davg_at_100 = []
            print(f"Optimized (At 100): Steps {step+1}, Loss: {avgloss100}, d_avg: {avgdavg100}")
            with open(avg_values_at_100, "a") as file:
                file.write(f"Optimized (At 100): Steps {step+1}, Loss: {avgloss100}, d_avg: {avgdavg100}\n")
            writer.add_scalar('Avg_loss_at_every_100', avgloss100, step+1)
            writer.add_scalar('Avg_d_avg_at_every_100', avgdavg100, step+1)
        if (step+1)%500 == 0:
            file_path = os.path.join(ckpt_dir, f'model_step_{step}.pth')
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, file_path)
            print("Saved model")

        step += 1


    writer.close()

train(model, optimizer,train_dataloader, load_model=True, new_lr = 5e-6)



