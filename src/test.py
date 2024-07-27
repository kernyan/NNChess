import torch
import torch.nn as nn
import torch.nn.functional as F

MODELS = "/untether/workspace/Boqueria/hackathon/external/NNChess/src/models"

import sys
sys.path.append(str(MODELS))

from resnet18_uai import RN18Model
#from resnet50_uai import RN50Model

MODEL_USED = RN18Model

with torch.no_grad():
    sample_input = torch.rand((1, 224, 224, 3))  # Example input with batch size of 1
    output = MODEL_USED().eval()(sample_input)  # Forward pass
    print(f'Output: {output}')
