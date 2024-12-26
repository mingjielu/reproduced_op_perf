import torch
import torch.nn as nn
from torch.fx import symbolic_trace

# 示例模型
class ExampleModel(nn.Module):
    def __init__(self):
        super(ExampleModel, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.add(x, x)  # 调用 torch.add
        x = self.fc2(x)
        return x

# 初始化模型
model = ExampleModel()
from IPython import embed;embed()
# 使用 FX 进行符号追踪
traced_model = symbolic_trace(model)

# 遍历捕获的操作
for node in traced_model.graph.nodes:
    if node.target == torch.add:
        print(f"Found torch.add: {node}")

