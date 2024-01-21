from .model import EncoderProcesserDecoder
import torch.nn as nn
import torch
from torch_geometric.data import Data
from utils import normalization
import os



class Simulator(nn.Module):

    def __init__(self, message_passing_num, node_input_size, edge_input_size, device) -> None:
        super(Simulator, self).__init__()

        self.node_input_size = node_input_size
        self.edge_input_size = edge_input_size
        self.model = EncoderProcesserDecoder(message_passing_num=message_passing_num, node_input_size=node_input_size, edge_input_size=edge_input_size).to(device)
        self._output_normalizer = normalization.Normalizer(size=2, name='output_normalizer', device=device)
        self._node_normalizer = normalization.Normalizer(size=node_input_size, name='node_normalizer', device=device)
        # self._edge_normalizer = normalization.Normalizer(size=edge_input_size, name='edge_normalizer', device=device)

        print('Simulator model initialized')

    def update_node_attr(self, frames, types:torch.Tensor):
        node_feature = []

        node_feature.append(frames) #velocity
        node_type = torch.squeeze(types.long())
        one_hot = torch.nn.functional.one_hot(node_type, 9)
        node_feature.append(one_hot)
        node_feats = torch.cat(node_feature, dim=1)
        normalized_node_feats = self._node_normalizer(node_feats, self.training)

        return normalized_node_feats

    def velocity_to_accelation(self, noised_frames, next_velocity):

        acc_next = next_velocity - noised_frames
        return acc_next


    def forward(self, graph:Data, velocity_sequence_noise):
        
        if self.training:
            
            node_type = graph.x[:, 0:1]
            frames = graph.x[:, 1:3]
            target = graph.y

            noised_frames = frames + velocity_sequence_noise
            node_attr = self.update_node_attr(noised_frames, node_type)
            graph.x = node_attr
            predicted = self.model(graph)

            target_acceration = self.velocity_to_accelation(noised_frames, target)
            target_acceration_normalized = self._output_normalizer(target_acceration, self.training)

            return predicted, target_acceration_normalized

        else:

            node_type = graph.x[:, 0:1]
            frames = graph.x[:, 1:3]
            node_attr = self.update_node_attr(frames, node_type)
            graph.x = node_attr
            predicted = self.model(graph)

            velocity_update = self._output_normalizer.inverse(predicted)
            predicted_velocity = frames + velocity_update

            return predicted_velocity

    def load_checkpoint(self, model_path):

        # load model state
        model_dicts = torch.load(model_path)
        self.load_state_dict(model_dicts['model'])

        # load train hyperparameters
        keys = list(model_dicts.keys())
        keys.remove('model')
        keys.remove('step')

        for k in keys:
            v = model_dicts[k]
            for para, value in v.items():
                object = eval('self.'+k)
                setattr(object, para, value)

        print(f"Load model at {model_path}")


    def save_checkpoint(self, step, optimizer_state, savedir=None):

        # Save model
        to_save_model = {
            'step': step,
            'model': self.state_dict(),
            '_output_normalizer': self._output_normalizer.get_variable(),
            '_node_normalizer': self._node_normalizer.get_variable()
            # _edge_normalizer = self._edge_normalizer.get_variable()
        }
        torch.save(to_save_model,
                   os.path.join(savedir, f'model-{step}.pt'))

        # Save optimizer state
        to_save_optimizer = {
            'step': step,
            'optimizer_state': optimizer_state,
        }
        torch.save(to_save_optimizer,
                   os.path.join(savedir, f'train_state-{step}.pt'))

        print(f"Checkpoint saved in {savedir}")
