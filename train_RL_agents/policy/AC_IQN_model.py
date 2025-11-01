import torch
import torch.nn as nn
import numpy as np
import os
import json
from torch.nn.functional import softmax,relu,dropout
import copy

def encoder(input_dimension,output_dimension):
    l1 = nn.Linear(input_dimension,output_dimension)
    l2 = nn.ReLU()
    model = nn.Sequential(l1, l2)
    return model


class AC_IQN_Policy():
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
                 seed=0
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
                           seed).to(device)
        self.critic = Critic(self_dimension,
                             object_dimension,
                             max_object_num,
                             self_feature_dimension,
                             object_feature_dimension,
                             concat_feature_dimension,
                             hidden_dimension,
                             len(value_ranges_of_action),
                             device,
                             seed+1).to(device)
        
    def save(self,directory):
        self.actor.save(directory)
        self.critic.save(directory)

    @classmethod
    def load(cls, directory, device="cpu"):
        # Load actor and critic parameters
        actor = Actor.load(directory, device)
        critic = Critic.load(directory, device)

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
        policy.critic = critic

        return policy
    

# class Actor(nn.Module):
#     def __init__(self,
#                  self_dimension,
#                  object_dimension,
#                  max_object_num,
#                  self_feature_dimension,
#                  object_feature_dimension,
#                  concat_feature_dimension,
#                  hidden_dimension,
#                  value_ranges_of_action,
#                  device='cpu',
#                  seed=0
#                  ):
#         super().__init__()

#         self.self_dimension = self_dimension
#         self.object_dimension = object_dimension
#         self.max_object_num = max_object_num
#         self.self_feature_dimension = self_feature_dimension
#         self.object_feature_dimension = object_feature_dimension
#         self.concat_feature_dimension = concat_feature_dimension
#         self.hidden_dimension = hidden_dimension
#         self.value_ranges_of_action = copy.deepcopy(value_ranges_of_action)
#         self.action_dimension = len(self.value_ranges_of_action)
#         self.device = device
#         self.seed_id = seed
#         self.seed = torch.manual_seed(seed)
#         self.compute_action_matrices()

#         self.K = 32 # number of quantiles in output
#         self.n = 64 # number of cosine features

#         # observation encoders
#         self.self_encoder = encoder(self_dimension,self_feature_dimension)
#         self.object_encoder = encoder(object_dimension,object_feature_dimension)

#         # quantile encoder
#         self.pis = torch.FloatTensor([np.pi * i for i in range(self.n)]).view(1,1,self.n).to(device)
#         self.cos_embedding = nn.Linear(self.n,self.concat_feature_dimension)

#         # hidden layers
#         self.hidden_layer = nn.Linear(self.concat_feature_dimension, hidden_dimension)
#         self.hidden_layer_2 = nn.Linear(hidden_dimension, hidden_dimension)
        
#         # output layers
#         self.output_layer = nn.Linear(hidden_dimension, self.action_dimension)

#     def compute_action_matrices(self):
#         value_ranges_of_action = torch.tensor(self.value_ranges_of_action)
#         self.atan_scale = torch.tensor(2.0/torch.pi).to(self.device)
#         self.action_amplitude = 0.5*(value_ranges_of_action[:,1]-value_ranges_of_action[:,0])
#         self.action_amplitude = torch.diag(self.action_amplitude).to(self.device)
#         self.action_mean = torch.mean(value_ranges_of_action,dim=1).to(self.device)

#     def calc_cos(self, batch_size, num_tau=8, cvar=1.0):
#         """
#         Calculating the cosinus values depending on the number of tau samples
#         """
#         # temp for reproducibility
#         # taus = torch.tensor([[0.05, 0.08, 0.11, 0.14, 0.17, 0.2 , 0.23, 0.26, 0.29, 0.32, 0.35, \
#         #                     0.38, 0.41, 0.44, 0.47, 0.5 , 0.53, 0.56, 0.59, 0.62, 0.65, 0.68, \
#         #                     0.71, 0.74, 0.77, 0.8 , 0.83, 0.86, 0.89, 0.92, 0.95, 0.98]]).to(self.device).unsqueeze(-1)

#         taus = torch.rand(batch_size,num_tau).to(self.device).unsqueeze(-1)

#         # distorted quantile sampling
#         taus = taus * cvar

#         cos = torch.cos(taus * self.pis)
#         assert cos.shape == (batch_size, num_tau, self.n), "cos shape is incorrect"
#         return cos, taus
    
#     def observation_processor(self, x, num_tau=8, cvar=1.0):
#         assert len(x) == 3, "The number of elements in state must be 3!"
#         x_1, x_2, x_2_mask = x

#         batch_size = x_1.shape[0]

#         # self state features batch
#         x_1 = self.self_encoder(x_1)
        
#         if x_2 is None:
#             x_2 = torch.zeros((batch_size,self.max_object_num*self.object_feature_dimension)).to(self.device)
#         else:
#             # encode object observations
#             x_2 = x_2.view(batch_size*self.max_object_num,self.object_dimension)
#             x_2 = self.object_encoder(x_2)
#             x_2 = x_2.view(batch_size,self.max_object_num,self.object_feature_dimension)

#             # apply object mask to padding
#             x_2 = x_2.masked_fill(x_2_mask.unsqueeze(-1)<0.5,0.0)

#             x_2 = x_2.view(batch_size,self.max_object_num*self.object_feature_dimension)

#         features=torch.cat((x_1,x_2),1)

#         # encode quantiles as features
#         cos, taus = self.calc_cos(batch_size, num_tau, cvar)
#         cos = cos.view(batch_size*num_tau, self.n)
#         cos_features = relu(self.cos_embedding(cos)).view(batch_size,num_tau,self.concat_feature_dimension)

#         # pairwise product of the input feature and cosine features
#         features = (features.unsqueeze(1) * cos_features).view(batch_size*num_tau,self.concat_feature_dimension)
        
#         return features, taus

#     def forward(self, x, num_tau=8, cvar=1.0):
#         assert len(x) == 3, "The number of elements in state must be 3!"
#         x_1,_,_ = x
#         batch_size = x_1.shape[0]

#         features, taus = self.observation_processor(x,num_tau,cvar)

#         features = relu(self.hidden_layer(features))
#         features = relu(self.hidden_layer_2(features))
        
#         action_quantiles = self.output_layer(features)
#         action_quantiles = action_quantiles.view(batch_size,num_tau,self.action_dimension)
#         actions = action_quantiles.mean(dim=1)

#         # map to value ranges of action
#         # actions = (self.atan_scale*torch.atan(actions)) @ self.action_amplitude + self.action_mean
#         actions = self.atan_scale*torch.atan(actions)

#         return actions, taus
    
#     def get_constructor_parameters(self):       
#         return dict(self_dimension = self.self_dimension,
#                     object_dimension = self.object_dimension,
#                     max_object_num = self.max_object_num,
#                     self_feature_dimension = self.self_feature_dimension,
#                     object_feature_dimension = self.object_feature_dimension,
#                     concat_feature_dimension = self.concat_feature_dimension,
#                     hidden_dimension = self.hidden_dimension,
#                     value_ranges_of_action = self.value_ranges_of_action,
#                     seed = self.seed_id)
    
#     def save(self,directory):
#         # save network parameters
#         torch.save(self.state_dict(),os.path.join(directory,f"actor_network_params.pth"))
        
#         # save constructor parameters
#         with open(os.path.join(directory,f"actor_constructor_params.json"),mode="w") as constructor_f:
#             json.dump(self.get_constructor_parameters(),constructor_f)

#     @classmethod
#     def load(cls,directory,device="cpu"):
#         # load network parameters
#         model_params = torch.load(os.path.join(directory,"actor_network_params.pth"),
#                                   map_location=device)

#         # load constructor parameters
#         with open(os.path.join(directory,"actor_constructor_params.json"), mode="r") as constructor_f:
#             constructor_params = json.load(constructor_f)
#             constructor_params["device"] = device

#         model = cls(**constructor_params)
#         model.load_state_dict(model_params)
#         model.to(device)

#         return model
    
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
        self.value_ranges_of_action = copy.deepcopy(value_ranges_of_action)
        self.action_dimension = len(self.value_ranges_of_action)
        self.device = device
        self.seed_id = seed
        self.seed = torch.manual_seed(seed)
        
        self.atan_scale = torch.tensor(2.0/torch.pi).to(self.device)

        # observation encoders
        self.self_encoder = encoder(self_dimension,self_feature_dimension)
        self.object_encoder = encoder(object_dimension,object_feature_dimension)

        # hidden layers
        self.hidden_layer = nn.Linear(self.concat_feature_dimension, hidden_dimension)
        self.hidden_layer_2 = nn.Linear(hidden_dimension, hidden_dimension)
        
        # output layers
        self.output_layer = nn.Linear(hidden_dimension, self.action_dimension)
    
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

    def forward(self, x):
        assert len(x) == 3, "The number of elements in state must be 3!"

        features = self.observation_processor(x)

        features = relu(self.hidden_layer(features))
        features = relu(self.hidden_layer_2(features))
        
        actions = self.output_layer(features)

        # map to (-1,1)
        actions = self.atan_scale*torch.atan(actions)

        return actions
    
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

        self.K = 32 # number of quantiles in output
        self.n = 64 # number of cosine features

        # observation encoders
        self.self_encoder = encoder(self_dimension,self_feature_dimension)
        self.object_encoder = encoder(object_dimension,object_feature_dimension)

        # quantile encoder
        self.pis = torch.FloatTensor([np.pi * i for i in range(self.n)]).view(1,1,self.n).to(device)
        self.cos_embedding = nn.Linear(self.n,self.concat_feature_dimension)

        # action encoder
        self.action_encoder = encoder(self.action_dimension,hidden_dimension)

        # hidden layers
        self.hidden_layer = nn.Linear(self.concat_feature_dimension, hidden_dimension)
        self.hidden_layer_2 = nn.Linear(hidden_dimension, hidden_dimension)
        
        # output layers
        self.output_layer = nn.Linear(hidden_dimension, 1)

    def calc_cos(self, batch_size, num_tau=8, cvar=1.0):
        """
        Calculating the cosinus values depending on the number of tau samples
        """
        # temp for reproducibility
        # taus = torch.tensor([[0.05, 0.08, 0.11, 0.14, 0.17, 0.2 , 0.23, 0.26, 0.29, 0.32, 0.35, \
        #                     0.38, 0.41, 0.44, 0.47, 0.5 , 0.53, 0.56, 0.59, 0.62, 0.65, 0.68, \
        #                     0.71, 0.74, 0.77, 0.8 , 0.83, 0.86, 0.89, 0.92, 0.95, 0.98]]).to(self.device).unsqueeze(-1)

        taus = torch.rand(batch_size,num_tau).to(self.device).unsqueeze(-1)

        # distorted quantile sampling
        taus = taus * cvar

        cos = torch.cos(taus * self.pis)
        assert cos.shape == (batch_size, num_tau, self.n), "cos shape is incorrect"
        return cos, taus
    
    def observation_processor(self, x, num_tau=8, cvar=1.0):
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

        # encode quantiles as features
        cos, taus = self.calc_cos(batch_size, num_tau, cvar)
        cos = cos.view(batch_size*num_tau, self.n)
        cos_features = relu(self.cos_embedding(cos)).view(batch_size,num_tau,self.concat_feature_dimension)

        # pairwise product of the input feature and cosine features
        features = (features.unsqueeze(1) * cos_features).view(batch_size*num_tau,self.concat_feature_dimension)
        
        return features, taus
    
    def forward(self, x, actions, num_tau=8, cvar=1.0):
        assert len(x) == 3, "The number of elements in state must be 3!"
        x_1,_,_ = x
        batch_size = x_1.shape[0]

        features, taus = self.observation_processor(x,num_tau,cvar)

        features = relu(self.hidden_layer(features))

        # encode action
        action_features = self.action_encoder(actions)
        features = features.view(batch_size,num_tau,self.hidden_dimension)
        features = (action_features.unsqueeze(1) * features).view(batch_size*num_tau,self.hidden_dimension)

        features = relu(self.hidden_layer_2(features))

        quantiles = self.output_layer(features)

        return quantiles.view(batch_size,num_tau), taus
    
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
    
    def save(self,directory):
        # save network parameters
        torch.save(self.state_dict(),os.path.join(directory,f"critic_network_params.pth"))
        
        # save constructor parameters
        with open(os.path.join(directory,f"critic_constructor_params.json"),mode="w") as constructor_f:
            json.dump(self.get_constructor_parameters(),constructor_f)

    @classmethod
    def load(cls,directory,device="cpu"):
        # load network parameters
        model_params = torch.load(os.path.join(directory,"critic_network_params.pth"),
                                  map_location=device)

        # load constructor parameters
        with open(os.path.join(directory,"critic_constructor_params.json"), mode="r") as constructor_f:
            constructor_params = json.load(constructor_f)
            constructor_params["device"] = device

        model = cls(**constructor_params)
        model.load_state_dict(model_params)
        model.to(device)

        return model