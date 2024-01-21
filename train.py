from dataset import FPC
from model.simulator import Simulator
import torch
from utils.noise import get_velocity_noise
import time
from utils.utils import NodeType
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
import os
import glob
import re
from absl import flags
from absl import app


flags.DEFINE_integer('batch_size', 2, help='The batch size.')
flags.DEFINE_string('data_path', '/work2/08264/baagee/frontera/gns-meshnet-data/gns-data/datasets/pipe-h5/', help='The dataset directory.')
flags.DEFINE_string('model_path', '/work2/08264/baagee/frontera/gns-meshnet-data/gns-data/models/pipe-h5/', help=('The path for saving checkpoints of the model.'))
flags.DEFINE_string('model_file', None, help=('Model filename (.pt) to resume from. Can also use "latest" to default to newest file.'))
flags.DEFINE_string('train_state_file', None, help=('Train state filename (.pt) to resume from. Can also use "latest" to default to newest file.'))
flags.DEFINE_integer('ntraining_steps', int(2E7), help='Number of training steps.')
flags.DEFINE_integer('nsave_steps', int(1000), help='Number of steps at which to save the model.')
flags.DEFINE_integer('nprint_steps', int(10), help='Number of steps at which to print the model.')

# Learning rate parameters
flags.DEFINE_float('lr_init', 1e-4, help='Initial learning rate.')
flags.DEFINE_float('lr_decay', 0.1, help='Learning rate decay.')
flags.DEFINE_integer('lr_decay_steps', int(5e6), help='Learning rate decay steps.')

FLAGS = flags.FLAGS


batch_size = 2
noise_std = 2e-2

print_batch = 10
save_batch = 1000

transformer = T.Compose([T.FaceToEdge(), T.Cartesian(norm=False), T.Distance(norm=False)])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(simulator: Simulator, dataloader, optimizer):

    step = 0

    # If model_path does not exist create new directory.
    if not os.path.exists(FLAGS.model_path):
        os.makedirs(FLAGS.model_path)

    # If model_path does exist and model_file and train_state_file exist continue training.
    if FLAGS.model_file is not None:

        if FLAGS.model_file == "latest" and FLAGS.train_state_file == "latest":
            # find the latest model, assumes model and train_state files are in step.
            fnames = glob.glob(f'{FLAGS.model_path}*model*pt')
            max_model_number = 0
            expr = re.compile(".*model-(\d+).pt")
            for fname in fnames:
                model_num = int(expr.search(fname).groups()[0])
                if model_num > max_model_number:
                    max_model_number = model_num
            # reset names to point to the latest.
            FLAGS.model_file = f"model-{max_model_number}.pt"
            FLAGS.train_state_file = f"train_state-{max_model_number}.pt"

        if os.path.exists(FLAGS.model_path + FLAGS.model_file) and os.path.exists(
                FLAGS.model_path + FLAGS.train_state_file):
            # load model
            simulator.load_checkpoint(FLAGS.model_path + FLAGS.model_file)

            # load train state
            train_state = torch.load(FLAGS.model_path + FLAGS.train_state_file)
            # set optimizer state
            optimizer = torch.optim.Adam(simulator.parameters())
            optimizer.load_state_dict(train_state["optimizer_state"])
            # set global train state
            step = train_state.pop("step")

        else:
            msg = f'Specified model_file {FLAGS.model_path + FLAGS.model_file} and train_state_file {FLAGS.model_path + FLAGS.train_state_file} not found.'
            raise FileNotFoundError(msg)

    simulator.train()
    simulator.to(device)

    not_reached_nsteps = True
    try:
        while not_reached_nsteps:
            for _, graph in enumerate(dataloader):

                graph = transformer(graph)
                graph = graph.cuda()

                node_type = graph.x[:, 0] #"node_type, cur_v, pressure, time"
                velocity_sequence_noise = get_velocity_noise(graph, noise_std=noise_std, device=device)
                predicted_acc, target_acc = simulator(graph, velocity_sequence_noise)
                mask = torch.logical_or(node_type==NodeType.NORMAL, node_type==NodeType.OUTFLOW)

                errors = ((predicted_acc - target_acc)**2)[mask]
                loss = torch.mean(errors)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update learning rate
                lr_new = FLAGS.lr_init * (FLAGS.lr_decay ** (step / FLAGS.lr_decay_steps)) + 1e-6
                for param in optimizer.param_groups:
                    param['lr'] = lr_new

                if step % FLAGS.nprint_steps == 0:
                    print(f'Step {step}, loss {loss.item()}')

                if step % FLAGS.nsave_steps == 0:
                    simulator.save_checkpoint(
                        step, optimizer.state_dict(), savedir=FLAGS.model_path)

                # Complete training
                if (step >= FLAGS.ntraining_steps):
                    not_reached_nsteps = False
                    break

                step += 1

    except KeyboardInterrupt:
        pass

def main(_):

    # Init simulator and optimizer
    simulator = Simulator(message_passing_num=10, node_input_size=11, edge_input_size=3, device=device)
    # train_state = torch.load("/work2/08264/baagee/frontera/meshnet/checkpoint/simulator.pth")
    optimizer = torch.optim.Adam(simulator.parameters(), lr=1e-4)

    dataset_fpc = FPC(dataset_dir=FLAGS.data_path, split='train', max_epochs=50)
    train_loader = DataLoader(dataset=dataset_fpc, batch_size=batch_size, num_workers=10)
    train(simulator, train_loader, optimizer)

if __name__ == '__main__':
    app.run(main)
