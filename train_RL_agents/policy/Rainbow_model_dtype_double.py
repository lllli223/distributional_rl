# Based on (https://github.com/Kaixhin/Rainbow/blob/master/model.py)

import math
import torch
from torch import nn
from torch.nn import functional as F
import os
import json

def encoder(input_dimension,output_dimension):
    l1 = nn.Linear(input_dimension,output_dimension,dtype=torch.float64)
    l2 = nn.ReLU()
    model = nn.Sequential(l1, l2)
    return model

# Factorized NoisyLinear layer with bias
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.05):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float64))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float64))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features, dtype=torch.float64))
        self.bias_mu = nn.Parameter(torch.empty(out_features, dtype=torch.float64))
        self.bias_sigma = nn.Parameter(torch.empty(out_features, dtype=torch.float64))
        self.register_buffer('bias_epsilon', torch.empty(out_features, dtype=torch.float64))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size, dtype=torch.float64, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input):
        return F.linear(input, self.weight_mu, self.bias_mu)


class Rainbow_Policy(nn.Module):
    def __init__(self,
                self_dimension,
                object_dimension,
                max_object_num,
                self_feature_dimension,
                object_feature_dimension,
                concat_feature_dimension,
                hidden_dimension,
                action_size,
                atoms,
                device='cpu',
                seed=0):
        super().__init__()

        self.self_dimension = self_dimension
        self.object_dimension = object_dimension
        self.max_object_num = max_object_num
        self.self_feature_dimension = self_feature_dimension
        self.object_feature_dimension = object_feature_dimension
        self.concat_feature_dimension = concat_feature_dimension
        self.hidden_dimension = hidden_dimension
        self.action_size = action_size
        self.atoms = atoms
        self.device = device
        self.seed_id = seed
        self.seed = torch.manual_seed(seed)

        self.self_encoder = encoder(self_dimension,self_feature_dimension)
        self.object_encoder = encoder(object_dimension,object_feature_dimension)

        # hidden layers
        self.hidden_layer_v = NoisyLinear(self.concat_feature_dimension, hidden_dimension)
        self.hidden_layer_a = NoisyLinear(self.concat_feature_dimension, hidden_dimension)

        self.hidden_layer_v_2 = NoisyLinear(hidden_dimension, hidden_dimension)
        self.hidden_layer_a_2 = NoisyLinear(hidden_dimension, hidden_dimension)

        self.output_layer_v = NoisyLinear(hidden_dimension, self.atoms)
        self.output_layer_a = NoisyLinear(hidden_dimension, action_size * self.atoms)

    def forward(self, x, log=False):
        assert len(x) == 3, "The number of elements in state must be 3!"
        x_1, x_2, x_2_mask = x

        batch_size = x_1.shape[0]

        # self state features batch
        x_1 = self.self_encoder(x_1)
    
        if x_2 is None:
            x_2 = torch.zeros((batch_size,self.max_object_num*self.object_feature_dimension)).to(dtype=torch.float64).to(self.device)
        else:
            # encode object observations
            x_2 = x_2.view(batch_size*self.max_object_num,self.object_dimension)
            x_2 = self.object_encoder(x_2)
            x_2 = x_2.view(batch_size,self.max_object_num,self.object_feature_dimension)

            # apply object mask to padding
            x_2 = x_2.masked_fill(x_2_mask.unsqueeze(-1)<0.5,0.0).to(dtype=torch.float64)

            x_2 = x_2.view(batch_size,self.max_object_num*self.object_feature_dimension)

        features=torch.cat((x_1,x_2),1)

        # Value stream
        features_v = F.relu(self.hidden_layer_v(features))
        features_v = F.relu(self.hidden_layer_v_2(features_v))
        v = self.output_layer_v(features_v)  

        # Advantage stream
        features_a = F.relu(self.hidden_layer_a(features))
        features_a = F.relu(self.hidden_layer_a_2(features_a))
        a = self.output_layer_a(features_a)  
        
        v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_size, self.atoms)
        q = v + a - a.mean(1, keepdim=True)  # Combine streams

        if log:  # Use log softmax for numerical stability
            q = F.log_softmax(q, dim=2)  # Log probabilities with action over second dimension
        else:
            q = F.softmax(q, dim=2)  # Probabilities with action over second dimension
        
        return q

    def reset_noise(self):
        for name, module in self.named_children():
            if "hidden_layer" in name or "output_layer" in name:
                module.reset_noise()

    def get_constructor_parameters(self):       
        return dict(self_dimension = self.self_dimension,
                    object_dimension = self.object_dimension,
                    max_object_num = self.max_object_num,
                    self_feature_dimension = self.self_feature_dimension,
                    object_feature_dimension = self.object_feature_dimension,
                    concat_feature_dimension = self.concat_feature_dimension,
                    hidden_dimension = self.hidden_dimension,
                    action_size = self.action_size,
                    atoms = self.atoms,
                    seed = self.seed_id)
    
    def save(self,directory):
        # save network parameters
        torch.save(self.state_dict(),os.path.join(directory,f"network_params.pth"))
        
        # save constructor parameters
        with open(os.path.join(directory,f"constructor_params.json"),mode="w") as constructor_f:
            json.dump(self.get_constructor_parameters(),constructor_f)

    @classmethod
    def load(cls,directory,device="cpu"):
        # load network parameters
        model_params = torch.load(os.path.join(directory,"network_params.pth"),
                                  map_location=device)

        # load constructor parameters
        with open(os.path.join(directory,"constructor_params.json"), mode="r") as constructor_f:
            constructor_params = json.load(constructor_f)
            constructor_params["device"] = device

        model = cls(**constructor_params)
        model.load_state_dict(model_params)
        model.to(device)

        return model

