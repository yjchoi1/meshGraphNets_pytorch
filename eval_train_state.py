import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle
import time
import itertools
import sys
import os
import pandas as pd

# sys.path.append('/work2/08264/baagee/frontera/meshnet/utils/')
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from dataset import FPC
from model.simulator import Simulator
from utils.noise import get_velocity_noise
from utils.utils import NodeType


batch_size = 2
noise_std = 2e-2
node_type_embedding_size = 9
dt = 0.01
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transformer = T.Compose([T.FaceToEdge(), T.Cartesian(norm=False), T.Distance(norm=False)])
# eval_steps = np.append(np.arange(0, 1000001, 5000), np.arange(1000000, 5000001, 10000))
eval_steps = np.arange(0, 2000000, 5000)
nexamples_for_loss = 10

data_name = "pipe-h5"
data_path = f"/work2/08264/baagee/frontera/gns-meshnet-data/gns-data/datasets/{data_name}/"
model_path = f"/work2/08264/baagee/frontera/gns-meshnet-data/gns-data/models/{data_name}/"
output_path = f"/work2/08264/baagee/frontera/gns-meshnet-data/gns-data/models/{data_name}/"


@torch.no_grad()
def get_loss_history(eval_steps, dataset, nexamples_to_eval):

    # Init simulator
    simulator = Simulator(
        message_passing_num=10, node_input_size=11, edge_input_size=3, device=device)

    loss_hist = []
    # loop over the training steps and evaluate the loss
    for step in eval_steps:
        # Load simulator
        simulator.load_checkpoint(model_path + f"model-{step}.pt")
        simulator.to(device)

        total_loss = []

        for _, graph in itertools.islice(enumerate(dataset), nexamples_to_eval):
            graph = transformer(graph)
            graph = graph.cuda()

            node_type = graph.x[:, 0]  # "node_type, cur_v, pressure, time"
            velocity_sequence_noise = get_velocity_noise(graph, noise_std=noise_std, device=device)

            # Predict
            predicted_acc, target_acc = simulator(graph, velocity_sequence_noise)
            mask = torch.logical_or(node_type == NodeType.NORMAL, node_type == NodeType.OUTFLOW)

            # Loss
            errors = ((predicted_acc - target_acc) ** 2)[mask]
            loss = torch.mean(errors).to("cpu").numpy()

            # Append loss
            total_loss.append(loss)

        # compute mean of the loss evaluated for 'nexamples'
        mean_loss = np.mean(total_loss)
        print(f"Mean loss evaluated with {nexamples_to_eval} samples at train step {step}: {mean_loss}")

        # append the mean loss for the current model with the current step
        loss_hist.append(mean_loss)

    loss_history = np.vstack((eval_steps, loss_hist))
    return loss_history


# Get train and valid data
dataset_fpc = FPC(dataset_dir=data_path, split='train', max_epochs=50)
train_loader = DataLoader(dataset=dataset_fpc, batch_size=batch_size, num_workers=10)

dataset_fpc = FPC(dataset_dir=data_path, split='valid', max_epochs=50)
valid_loader = DataLoader(dataset=dataset_fpc, batch_size=batch_size, num_workers=10)

# Evaluate train loss
train_history = get_loss_history(
    eval_steps=eval_steps,
    dataset=train_loader,
    nexamples_to_eval=nexamples_for_loss)
# Save loss history
with open(f"{output_path}/train_history.pkl", 'wb') as f:
    pickle.dump(np.transpose(train_history), f)

# Evaluate validation loss
valid_history = get_loss_history(
    eval_steps=eval_steps,
    dataset=valid_loader,
    nexamples_to_eval=nexamples_for_loss)
# Save loss history
with open(f"{output_path}/valid_history.pkl", 'wb') as f:
    pickle.dump(np.transpose(valid_history), f)

#%% Load
with open(f"{output_path}/train_history.pkl", 'rb') as f:
    train_history_data = pickle.load(f)
with open(f"{output_path}/valid_history.pkl", 'rb') as f:
    valid_history_data = pickle.load(f)


# Convert loss data to pandas dataframe for data processing
val_df = pd.DataFrame(valid_history_data, columns=["step", "loss"])
train_df = pd.DataFrame(train_history_data, columns=["step", "loss"])

# Calculate moving average
sample_interval = 1
window_val = 2
window_train = 2
val_rolling_mean = val_df["loss"].rolling(window=window_val, center=True).mean()
train_rolling_mean = train_df["loss"].rolling(window=window_train, center=True).mean()

# Plot
fig, ax = plt.subplots(figsize=(5, 3))
# ax.plot(train_history_data[:, 0], train_history_data[:, 1], lw=1, alpha=0.5, label="train")
# ax.plot(valid_history_data[:, 0], valid_history_data[:, 1], alpha=0.5, label="Validation")
ax.plot(valid_history_data[:, 0], val_rolling_mean, alpha=0.5, label="Validation")
ax.plot(train_history_data[:, 0], train_rolling_mean, alpha=0.5, label="Training")
ax.set_yscale('log')
ax.set_xlabel("Steps")
ax.set_ylabel("Loss")
ax.set_xlim([0, 5000000])
ax.legend()
plt.tight_layout()
plt.savefig(f'{output_path}/loss_hist.png')
plt.show()
