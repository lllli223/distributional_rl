# Based on (https://github.com/thu-ml/tianshou/blob/master/tianshou/policy/modelfree/sac.py)

import torch
import torch.nn as nn
import numpy as np
import os
import json
from torch.nn.functional import softmax,relu,dropout
from torch.distributions import Normal
import copy

def encoder(input_dimension,output_dimension):
    l1 = nn.Linear(input_dimension,output_dimension,dtype=torch.float64)
    l2 = nn.ReLU()
    model = nn.Sequential(l1, l2)
    return model


class SAC_Policy():
    def __init__(self,
                 self_dimension,
                 object_dimension,
                 max_object_num,
                 self_feature_dimension,
                 object_feature_dimension,
                 concat_feature_dimension,
                 hidden_dimension,
                 value_ranges_of_action,
                 device='cpu',
                 seed=0,
                 min_log_std=-20,
                 max_log_std=2
                 ):
        
        self.actor = Actor(self_dimension,
                           object_dimension,
                           max_object_num,
                           self_feature_dimension,
                           object_feature_dimension,
                           concat_feature_dimension,
                           hidden_dimension,
                           value_ranges_of_action,
                           device,
                           seed,
                           min_log_std,
                           max_log_std).to(device)
        self.critic_1 = Critic(self_dimension,
                             object_dimension,
                             max_object_num,
                             self_feature_dimension,
                             object_feature_dimension,
                             concat_feature_dimension,
                             hidden_dimension,
                             len(value_ranges_of_action),
                             device,
                             seed+1).to(device)
        self.critic_2 = Critic(self_dimension,
                             object_dimension,
                             max_object_num,
                             self_feature_dimension,
                             object_feature_dimension,
                             concat_feature_dimension,
                             hidden_dimension,
                             len(value_ranges_of_action),
                             device,
                             seed+2).to(device)
        
    def save(self,directory):
        self.actor.save(directory)
        self.critic_1.save(1, directory)
        self.critic_2.save(2, directory)

    @classmethod
    def load(cls, directory, device="cpu"):
        # Load actor and critic parameters
        actor = Actor.load(directory, device)
        critic_1 = Critic.load(1, directory, device)
        critic_2 = Critic.load(2, directory, device)

        # Create an instance of the policy
        policy = cls(
            actor.self_dimension,
            actor.object_dimension,
            actor.max_object_num,
            actor.self_feature_dimension,
            actor.object_feature_dimension,
            actor.concat_feature_dimension,
            actor.hidden_dimension,
            actor.value_ranges_of_action,
            device=device
        )

        # Set the loaded actor and critic networks
        policy.actor = actor
        policy.critic_1 = critic_1
        policy.critic_2 = critic_2

        return policy
    

class Actor(nn.Module):
    def __init__(self,
                 self_dimension,
                 object_dimension,
                 max_object_num,
                 self_feature_dimension,
                 object_feature_dimension,
                 concat_feature_dimension,
                 hidden_dimension,
                 value_ranges_of_action,
                 device='cpu',
                 seed=0,
                 min_log_std=-20,
                 max_log_std=2
                 ):
        super().__init__()

        self.self_dimension = self_dimension
        self.object_dimension = object_dimension
        self.max_object_num = max_object_num
        self.self_feature_dimension = self_feature_dimension
        self.object_feature_dimension = object_feature_dimension
        self.concat_feature_dimension = concat_feature_dimension
        self.hidden_dimension = hidden_dimension
        self.value_ranges_of_action = copy.deepcopy(value_ranges_of_action)
        self.action_dimension = len(self.value_ranges_of_action)
        self.device = device
        self.seed_id = seed
        self.seed = torch.manual_seed(seed)
        
        self.atan_scale = torch.tensor(2.0/torch.pi,dtype=torch.float64).to(self.device)
        self.min_val = torch.tensor(1e-6,dtype=torch.float64).to(self.device)

        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

        # observation encoders
        self.self_encoder = encoder(self_dimension,self_feature_dimension)
        self.object_encoder = encoder(object_dimension,object_feature_dimension)

        # hidden layers
        self.hidden_layer = nn.Linear(self.concat_feature_dimension, hidden_dimension,dtype=torch.float64)
        self.hidden_layer_2 = nn.Linear(hidden_dimension, hidden_dimension,dtype=torch.float64)
        
        # output layers
        self.output_layer_mu = nn.Linear(hidden_dimension, self.action_dimension,dtype=torch.float64)
        self.output_layer_log_std = nn.Linear(hidden_dimension, self.action_dimension,dtype=torch.float64)
    
    def observation_processor(self, x):
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
        
        return features

    def forward(self, x):
        assert len(x) == 3, "The number of elements in state must be 3!"

        features = self.observation_processor(x)

        features = relu(self.hidden_layer(features))
        features = relu(self.hidden_layer_2(features))
        
        mu = self.output_layer_mu(features)
        log_std = self.output_layer_log_std(features)
        log_std = torch.clamp(log_std,self.min_log_std,self.max_log_std)

        return mu, log_std
    
    def get_constructor_parameters(self):       
        return dict(self_dimension = self.self_dimension,
                    object_dimension = self.object_dimension,
                    max_object_num = self.max_object_num,
                    self_feature_dimension = self.self_feature_dimension,
                    object_feature_dimension = self.object_feature_dimension,
                    concat_feature_dimension = self.concat_feature_dimension,
                    hidden_dimension = self.hidden_dimension,
                    value_ranges_of_action = self.value_ranges_of_action,
                    seed = self.seed_id)
    
    def save(self,directory):
        # save network parameters
        torch.save(self.state_dict(),os.path.join(directory,f"actor_network_params.pth"))
        
        # save constructor parameters
        with open(os.path.join(directory,f"actor_constructor_params.json"),mode="w") as constructor_f:
            json.dump(self.get_constructor_parameters(),constructor_f)

    @classmethod
    def load(cls,directory,device="cpu"):
        # load network parameters
        model_params = torch.load(os.path.join(directory,"actor_network_params.pth"),
                                  map_location=device)

        # load constructor parameters
        with open(os.path.join(directory,"actor_constructor_params.json"), mode="r") as constructor_f:
            constructor_params = json.load(constructor_f)
            constructor_params["device"] = device

        model = cls(**constructor_params)
        model.load_state_dict(model_params)
        model.to(device)

        return model


class Critic(nn.Module):
    def __init__(self,
                 self_dimension,
                 object_dimension,
                 max_object_num,
                 self_feature_dimension,
                 object_feature_dimension,
                 concat_feature_dimension,
                 hidden_dimension,
                 action_dimension,
                 device='cpu',
                 seed=0
                 ):
        super().__init__()

        self.self_dimension = self_dimension
        self.object_dimension = object_dimension
        self.max_object_num = max_object_num
        self.self_feature_dimension = self_feature_dimension
        self.object_feature_dimension = object_feature_dimension
        self.concat_feature_dimension = concat_feature_dimension
        self.hidden_dimension = hidden_dimension
        self.action_dimension = action_dimension
        self.device = device
        self.seed_id = seed
        self.seed = torch.manual_seed(seed)

        # observation encoders
        self.self_encoder = encoder(self_dimension,self_feature_dimension)
        self.object_encoder = encoder(object_dimension,object_feature_dimension)

        # action encoder
        self.action_encoder = encoder(self.action_dimension,hidden_dimension)

        # hidden layers
        self.hidden_layer = nn.Linear(self.concat_feature_dimension, hidden_dimension)
        self.hidden_layer_2 = nn.Linear(hidden_dimension, hidden_dimension)
        
        # output layers
        self.output_layer = nn.Linear(hidden_dimension, 1)
    
    def observation_processor(self, x):
        assert len(x) == 3, "The number of elements in state must be 3!"
        x_1, x_2, x_2_mask = x

        batch_size = x_1.shape[0]

        # self state features batch
        x_1 = self.self_encoder(x_1)
        
        if x_2 is None:
            x_2 = torch.zeros((batch_size,self.max_object_num*self.object_feature_dimension)).to(self.device)
        else:
            # encode object observations
            x_2 = x_2.view(batch_size*self.max_object_num,self.object_dimension)
            x_2 = self.object_encoder(x_2)
            x_2 = x_2.view(batch_size,self.max_object_num,self.object_feature_dimension)

            # apply object mask to padding
            x_2 = x_2.masked_fill(x_2_mask.unsqueeze(-1)<0.5,0.0)

            x_2 = x_2.view(batch_size,self.max_object_num*self.object_feature_dimension)

        features=torch.cat((x_1,x_2),1)
        
        return features
    
    def forward(self, x, actions):
        assert len(x) == 3, "The number of elements in state must be 3!"

        features = self.observation_processor(x)

        features = relu(self.hidden_layer(features))

        # encode action
        action_features = self.action_encoder(actions)
        features = action_features * features

        features = relu(self.hidden_layer_2(features))

        qvalues = self.output_layer(features)

        return qvalues
    
    def get_constructor_parameters(self):       
        return dict(self_dimension = self.self_dimension,
                    object_dimension = self.object_dimension,
                    max_object_num = self.max_object_num,
                    self_feature_dimension = self.self_feature_dimension,
                    object_feature_dimension = self.object_feature_dimension,
                    concat_feature_dimension = self.concat_feature_dimension,
                    hidden_dimension = self.hidden_dimension,
                    action_dimension = self.action_dimension,
                    seed = self.seed_id)
    
    def save(self,id,directory):
        # save network parameters
        torch.save(self.state_dict(),os.path.join(directory,f"critic_{id}_params.pth"))
        
        # save constructor parameters
        with open(os.path.join(directory,f"critic_{id}_constructor_params.json"),mode="w") as constructor_f:
            json.dump(self.get_constructor_parameters(),constructor_f)

    @classmethod
    def load(cls,id,directory,device="cpu"):
        # load network parameters
        model_params = torch.load(os.path.join(directory,f"critic_{id}_params.pth"),
                                  map_location=device)

        # load constructor parameters
        with open(os.path.join(directory,f"critic_{id}_constructor_params.json"), mode="r") as constructor_f:
            constructor_params = json.load(constructor_f)
            constructor_params["device"] = device

        model = cls(**constructor_params)
        model.load_state_dict(model_params)
        model.to(device)

        return model