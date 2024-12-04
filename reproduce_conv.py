import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.allow_tf32 = False
bs = 2
configs = [
    {"input_shape": (bs, 128, 1024, 1024), "weight_shape": (128, 128, 3, 3)},  # Conv1
    {"input_shape": (bs, 256, 512, 512), "weight_shape": (256, 256, 3, 3)},  # Conv2
    {"input_shape": (bs, 512, 256, 256), "weight_shape": (512, 512, 3, 3)},  # Conv3
    {"input_shape": (bs, 512, 128, 128), "weight_shape": (512, 512, 3, 3)}   # Conv4
]
num_warmup = 5
num_runs = 11
for config in configs:
    input_shape = config["input_shape"]
    weight_shape = config["weight_shape"]

    input_tensor = torch.randn(input_shape, dtype=torch.float32).to(device)

    conv_layer = torch.nn.Conv2d(in_channels=weight_shape[1], out_channels=weight_shape[0],
                                 kernel_size=3, stride=1, padding=1)
    conv_layer = conv_layer.to(device, dtype=torch.float32)  # 确保权重是 float32

    for i in range(num_warmup):
        output_tensor = conv_layer(input_tensor)

    total_time = 0.0
    for i in range(num_runs):
        start_time = time.time()
        torch.cuda.synchronize()
        output_tensor = conv_layer(input_tensor)
        torch.cuda.synchronize()
        end_time = time.time()
        if i >= 1:
            total_time += (end_time - start_time)

    average_time = total_time / (num_runs - 1)
    print(f"Average execution time for input {input_shape} and weight {weight_shape}: {average_time:.6f} seconds")
