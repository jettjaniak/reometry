from beartype.claw import beartype_this_package  # <-- hype comes
import torch
beartype_this_package()  # <-- hype comes
torch.set_grad_enabled(False)

__version__ = "2024.8.28"
