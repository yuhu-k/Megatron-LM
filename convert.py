import torch.nn as nn
import torch
from os import makedirs,listdir

hf_dir = 'llama-2-7b-hf'

bin_files = []

for file in listdir(hf_dir):
    if file.endswith('.bin'):
        bin_files.append(file)

machine = [{},{},{},{}]
feedforward = {}

for chpt in bin_files:

    state_dict = torch.load(hf_dir + '/' + chpt)

    for key in state_dict:
        if "rotary_emb" in key:
            continue
        if str(key) == "model.embed_tokens.weight":
            layer_num = 0
        elif key == "model.norm.weight" or key == "lm_head.weight":
            layer_num = 31
        else:
            layer_num = int(str(key).split('.')[2])
            
        if "gate_proj" in key:
            if feedforward.get(layer_num) is not None:
                output = torch.cat((state_dict[key],feedforward[layer_num]),dim=-1)
                key = f"model.layers.{layer_num}.mlp.linear_fc1.weight"
            else:
                feedforward[layer_num] = state_dict[key]
                continue
        elif "up_proj" in key:
            if feedforward.get(layer_num) is not None:
                output = torch.cat((feedforward[layer_num],state_dict[key]),dim=-1)
                key = f"model.layers.{layer_num}.mlp.linear_fc1.weight"
            else:
                feedforward[layer_num] = state_dict[key]
                continue
        else:
            output = state_dict[key]
            
        machine[int(layer_num/8)][key] = output
        print(key)

for i in range(machine.__len__()):
    makedirs(f'llama-2-7b-me/test-hf/iter_0000001/mp_rank_00_00{i}',exist_ok=True)
    torch.save(machine[i], f'llama-2-7b-me/test-hf/iter_0000001/mp_rank_00_00{i}/model_optim_rng.pt')