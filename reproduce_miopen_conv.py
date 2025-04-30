import torch
import time
from torch.profiler import profile, record_function, ProfilerActivity
#torch.backends.cudnn.allow_tf32 = False
configs = [
        {"input_shape":(4, 640, 100, 100), "weight_shape":(512, 640, 3, 3,), "stride_x": (6400000, 10000, 100, 1), "stride_w": (5760, 9, 3, 1), "dtype": torch.float32},
        {"input_shape":(4, 512, 100, 100), "weight_shape":(512, 512, 3, 3,), "stride_x": (5120000, 10000, 100, 1), "stride_w": (4608, 9, 3, 1), "dtype": torch.float32},
        #{"input_shape":(4, 512, 200, 100), "weight_shape":(512, 640, 3, 3,), "stride_x": (6400000, 10000, 100, 1), "stride_w": (5760, 9, 3, 1), "dtype": torch.float32},
        #{"input_shape":(4, 640, 100, 100), "weight_shape":(512, 640, 3, 3,), "stride_x": (6400000, 10000, 100, 1), "stride_w": (5760, 9, 3, 1), "dtype": torch.float32},
        ]
points = torch.randn(4,6,88,16,44,3,1,device='cuda',dtype=torch.float32)
post_rots = torch.randn(4,6,1,1,1,3,3,device='cuda',dtype=torch.float32)

for config in configs:
    input_tensor = torch.rand(config["input_shape"], dtype=config["dtype"]).to('cuda')
    conv_layer = torch.nn.Conv2d(in_channels=config["weight_shape"][1], out_channels=config["weight_shape"][0],
                                 kernel_size=config["weight_shape"][2], stride=1, padding=1).to('cuda')
    assert input_tensor.stride() == config["stride_x"], f'{input_tensor.stride()=}, {config["stride_x"]=}'
    assert conv_layer.weight.stride() == config["stride_w"], f'{conv_layer.weight.stride()=}, {config["stride_w"]=}'
    print('tensor/weight stride verified')
    for i in range(5): #warmup
        output_tensor = conv_layer(input_tensor)
        output_tensor.sum().backward()
    torch.cuda.synchronize()
    t1 = time.time()
    output_tensor = conv_layer(input_tensor)
    torch.cuda.synchronize()
    t2 = time.time()
    print('forward time(s/iter): ',t2-t1)
    output_tensor.sum().backward()
    torch.cuda.synchronize()
    t3 = time.time()
    print('backward time(s/iter): ',t3-t2)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True) as prof:
        output_tensor = conv_layer(input_tensor)
        output_tensor.sum().backward()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
