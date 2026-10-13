import os, torch, transformer_engine as te
import transformer_engine.pytorch as tep
from transformer_engine.common.recipe import Float8CurrentScaling, Format
lin = tep.Linear(1024, 1024, bias=False, params_dtype=torch.bfloat16).cuda()
x = torch.randn(64, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
with tep.fp8_autocast(enabled=True, fp8_recipe=Float8CurrentScaling(fp8_format=Format.HYBRID)):
    y = lin(x)
y.sum().backward()
print("OK  out", tuple(y.shape), y.dtype, "grad mean", float(x.grad.abs().mean()))
