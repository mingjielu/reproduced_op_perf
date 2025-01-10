import transformer_engine as te
import time
import torch

class TELayerNormColumnParallelLinear(te.pytorch.LayerNormLinear):
    def __init__(
        self,
        input_size: int,
        output_size: int,
    ):

        self.is_first_microbatch = True
        extra_kwargs = {}
        extra_kwargs['device'] = torch.cuda.current_device()
        extra_kwargs['params_dtype'] = torch.bfloat16

        extra_kwargs['normalization'] = 'RMSNorm'

        super().__init__(
            in_features=input_size,
            out_features=output_size,
            eps=1e-05,
            sequence_parallel=False,
            fuse_wgrad_accumulation=True,
            bias=False,
            return_bias=False,
            parallel_mode="column",
            return_layernorm_output=False,
            **extra_kwargs,
        )
#         print(f'in_features={input_size}, out_features={output_size}')
#         print(f'eps={self.config.layernorm_epsilon}')
#         print(f'sequence_parallel={self.config.sequence_parallel}')
#         print(f'te_group={get_tensor_model_parallel_group(check_initialized=False)}')
#         print(f'te_size={self.config.tensor_model_parallel_size}')
#         print(f'bias = {bias}')
#         print(f'return_bias = {self.te_return_bias}')
# in_features=5120, out_features=33792
# eps=1e-05
# sequence_parallel=False
# te_group=<torch.distributed.distributed_c10d.ProcessGroup object at 0x7ee6d3451470>
# te_size=1
# bias = False
# return_bias = False
# in_features=5120, out_features=6144
# eps=1e-05
# sequence_parallel=False
# te_group=<torch.distributed.distributed_c10d.ProcessGroup object at 0x7ee6d3451470>
# te_size=1
# bias = True
# return_bias = False

    def forward(self, x):
        out = super().forward(x, is_first_microbatch=self.is_first_microbatch)
        self.is_first_microbatch = False
        return out, None


LN = TELayerNormColumnParallelLinear(5120, 33792).cuda()
#LN = TELayerNormColumnParallelLinear(5120, 6144).cuda()

input = torch.randn([2048, 2, 5120], device='cuda', dtype=torch.bfloat16).requires_grad_(True)

out, _ = LN.forward(input)
#grad_output = torch.randn_like(out)
#out.backward(grad_output)
loss = out.sum()
loss.backward()
print(input.grad)



