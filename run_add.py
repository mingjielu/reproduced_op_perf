from reproduce_elementwise_add_profile import test_add
import torch

#custom_add = torch.add
#custom_add = torch.compile(torch.add)
#torch.add = custom_add

torch.add=torch.compile(torch.add)
test_add()
