import torch
import torch.nn as nn
import numpy as np
import os
import json
from torch.nn.functional import softmax,relu,dropout

def encoder(input_dimension,output_dimension):
    l1 = nn.Linear(input_dimension,output_dimension,dtype=torch.float64)
    l2 = nn.ReLU()
    model = nn.Sequential(l1, l2)
    return model

class IQN_Policy(nn.Module):
    def __init__(self,
                 self_dimension,
                 object_dimension,
                 max_object_num,
                 self_feature_dimension,
                 object_feature_dimension,
                 concat_feature_dimension,
                 hidden_dimension,
                 action_size,
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
        self.action_size = action_size
        self.device = device
        self.seed_id = seed
        self.seed = torch.manual_seed(seed)

        self.self_encoder = encoder(self_dimension,self_feature_dimension)
        self.object_encoder = encoder(object_dimension,object_feature_dimension)

        self.K = 32 # number of quantiles in output
        self.n = 64 # number of cosine features

        # quantile encoder
        self.pis = torch.tensor([np.pi * i for i in range(self.n)],dtype=torch.float64).view(1,1,self.n).to(device)
        self.cos_embedding = nn.Linear(self.n,self.concat_feature_dimension,dtype=torch.float64)

        # hidden layers
        self.hidden_layer = nn.Linear(self.concat_feature_dimension, hidden_dimension, dtype=torch.float64)
        self.hidden_layer_2 = nn.Linear(hidden_dimension, hidden_dimension, dtype=torch.float64)
        self.output_layer = nn.Linear(hidden_dimension, action_size, dtype=torch.float64)

    def calc_cos(self, taus):
        """
        Calculating the cosinus values depending on the number of tau samples
        """
        # temp for reproducibility
        # taus = torch.tensor([[0.05, 0.08, 0.11, 0.14, 0.17, 0.2 , 0.23, 0.26, 0.29, 0.32, 0.35, \
        #                     0.38, 0.41, 0.44, 0.47, 0.5 , 0.53, 0.56, 0.59, 0.62, 0.65, 0.68, \
        #                     0.71, 0.74, 0.77, 0.8 , 0.83, 0.86, 0.89, 0.92, 0.95, 0.98]], dtype=torch.float64).to(self.device).unsqueeze(-1)
        
        # taus = torch.rand(batch_size,num_tau).to(self.device).unsqueeze(-1)
        taus = taus.to(self.device).unsqueeze(-1)

        # distorted quantile sampling
        # taus = taus * cvar

        cos = torch.cos(taus * self.pis).to(dtype=torch.float64)
        # assert cos.shape == (batch_size, num_tau, self.n), "cos shape is incorrect"
        return cos
    
    def forward(self, x, taus):
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

        # encode quantiles as features
        cos = self.calc_cos(taus)
        cos = cos.view(batch_size*self.K, self.n)
        cos_features = relu(self.cos_embedding(cos)).view(batch_size,self.K,self.concat_feature_dimension)

        # pairwise product of the input feature and cosine features
        features = (features.unsqueeze(1) * cos_features).view(batch_size*self.K,self.concat_feature_dimension)
        
        features = relu(self.hidden_layer(features))
        features = relu(self.hidden_layer_2(features))
        quantiles = self.output_layer(features)
        
        return quantiles.view(batch_size,self.K,self.action_size)
    
    def get_constructor_parameters(self):       
        return dict(self_dimension = self.self_dimension,
                    object_dimension = self.object_dimension,
                    max_object_num = self.max_object_num,
                    self_feature_dimension = self.self_feature_dimension,
                    object_feature_dimension = self.object_feature_dimension,
                    concat_feature_dimension = self.concat_feature_dimension,
                    hidden_dimension = self.hidden_dimension,
                    action_size = self.action_size,
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

        # convert parameters to Double
        for param_name, param_tensor in model_params.items():
            model_params[param_name] = param_tensor.to(torch.float64)

        # load constructor parameters
        with open(os.path.join(directory,"constructor_params.json"), mode="r") as constructor_f:
            constructor_params = json.load(constructor_f)
            constructor_params["device"] = device

        model = cls(**constructor_params)
        model.load_state_dict(model_params)
        model.to(device)

        return model