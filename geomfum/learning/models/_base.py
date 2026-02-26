import abc

import torch.nn as nn


class BaseModel(abc.ABC, nn.Module):
    """Base class for all models."""
