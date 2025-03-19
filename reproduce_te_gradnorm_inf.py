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

    def forward(self, x):
        out = super().forward(x, is_first_microbatch=self.is_first_microbatch)
        self.is_first_microbatch = False
        return out, None


LN = TELayerNormColumnParallelLinear(5120, 33792).cuda()

for name,parameter in LN.named_parameters():
    print(name)
    print(parameter)
    parameter.main_grad=torch.zeros_like(parameter)
#LN = TELayerNormColumnParallelLinear(5120, 6144).cuda()

input = torch.randn([2048, 2, 5120], device='cuda', dtype=torch.bfloat16).requires_grad_(True)

out, _ = LN.forward(input)
grad_output = torch.randn_like(out)
out.backward(grad_output)
print(input.grad)



