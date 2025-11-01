from policy.IQN_model_dtype_double import IQN_Policy
from policy.AC_IQN_model_dtype_double import Actor as AC_IQN_actor
from policy.DDPG_model_dtype_double import Actor as DDPG_actor
from policy.DQN_model_dtype_double import DQN_Policy
from policy.SAC_model_dtype_double import Actor as SAC_actor
from policy.Rainbow_model_dtype_double import Rainbow_Policy
import torch
import os

model_type = "RL/agent/type"
device = "cpu"
load_dir = "directory/to/your/trained/RL/model"
save_dir = "torch/script/output/directory"
output_file = os.path.join(save_dir,f"traced_{model_type}_model.pt")

if model_type == "AC_IQN":
    AC_IQN_model = AC_IQN_actor.load(load_dir,device)

    example_input = ((torch.rand((1,7),dtype=torch.float64), \
                    torch.rand((1,5,5),dtype=torch.float64), \
                    torch.rand((1,5),dtype=torch.float64)), \
                    )
    traced_script_module = torch.jit.trace(AC_IQN_model, example_input)
    traced_script_module.save(output_file)
elif model_type == "IQN":
    IQN_model = IQN_Policy.load(load_dir,device)

    example_input = ((torch.rand((1,7),dtype=torch.float64), \
                    torch.rand((1,5,5),dtype=torch.float64), \
                    torch.rand((1,5),dtype=torch.float64)), \
                    torch.rand((1,32),dtype=torch.float64))
    traced_script_module = torch.jit.trace(IQN_model, example_input)
    traced_script_module.save(output_file)
elif model_type == "DDPG":
    DDPG_model = DDPG_actor.load(load_dir,device)

    example_input = ((torch.rand((1,7),dtype=torch.float64), \
                    torch.rand((1,5,5),dtype=torch.float64), \
                    torch.rand((1,5),dtype=torch.float64)), \
                    )
    traced_script_module = torch.jit.trace(DDPG_model, example_input)
    traced_script_module.save(output_file)
elif model_type == "DQN":
    DQN_model = DQN_Policy.load(load_dir,device)

    example_input = ((torch.rand((1,7),dtype=torch.float64), \
                    torch.rand((1,5,5),dtype=torch.float64), \
                    torch.rand((1,5),dtype=torch.float64)), \
                    )
    traced_script_module = torch.jit.trace(DQN_model, example_input)
    traced_script_module.save(output_file)
elif model_type == "SAC":
    SAC_model = SAC_actor.load(load_dir,device)

    example_input = ((torch.rand((1,7),dtype=torch.float64), \
                    torch.rand((1,5,5),dtype=torch.float64), \
                    torch.rand((1,5),dtype=torch.float64)), \
                    )
    traced_script_module = torch.jit.trace(SAC_model, example_input)
    traced_script_module.save(output_file)
elif model_type == "Rainbow":
    Rainbow_model = Rainbow_Policy.load(load_dir,device)

    example_input = ((torch.rand((1,7),dtype=torch.float64), \
                    torch.rand((1,5,5),dtype=torch.float64), \
                    torch.rand((1,5),dtype=torch.float64)), \
                    )
    traced_script_module = torch.jit.trace(Rainbow_model, example_input)
    traced_script_module.save(output_file)
