import torch
from semantic_dac import SemanticDAC
a = torch.zeros(1, 10000)
c = torch.randn(1, 20160)
model = SemanticDAC()
b = model.forward(c)
