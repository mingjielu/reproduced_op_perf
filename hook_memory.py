def register_weight_hooks(model):
    for name, module in model.named_modules():
        def make_hook(n, m):
            def hook(module, input, output):
                mem_allocated = torch.cuda.memory_allocated() / 1024**2
                mem_reserved  = torch.cuda.memory_reserved() / 1024**2
                print(f"{n}",
                      f"allocated={mem_allocated:.2f}MB, reserved={mem_reserved:.2f}MB")
            return hook
        module.register_forward_hook(make_hook(name, module)

    for name, param in model.named_parameters():
        if param.requires_grad:
            def make_hook(n):
                def hook(grad):
                    mem_allocated = torch.cuda.memory_allocated() / 1024**2
                    mem_reserved  = torch.cuda.memory_reserved() / 1024**2
                    print(f"[{n}] grad shape={tuple(grad.shape)}, "
                          f"allocated={mem_allocated:.2f}MB, reserved={mem_reserved:.2f}MB")
                return hook
            param.register_hook(make_hook(name))

register_hooks(model)
