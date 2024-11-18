import flag_gems as gems
import torch
gems.enable()

x = torch.randn([1024,1024], dtype=torch.float32, device=gems.device)

c = torch.add(x, x)
# with gems.use_gems():
#     pass

# with gems.device_guard(x.device):
#     torch.add(x, x)

print(x)
