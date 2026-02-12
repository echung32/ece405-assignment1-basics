import torch
from jaxtyping import Float
from torch import nn


class RMSNorm(nn.Module):
    """
    Following Touvron et al. [2023], we will use root mean square layer normalization
    (RMSNorm; Zhang and Sennrich, 2019, equation 4) for layer normalization
    """
    def __init__(self, d_model: int, eps: float = 1e-5, device = None, dtype=None) -> None:
        """
        Construct the RMSNorm module.

        Args:
            d_model (int): Hidden dimension of the model
            eps (float): Epsilon value for numerical stability
            device: torch.device | None - Device to store the parameters on
            dtype: torch.dtype | None - Data type of the parameters
        """
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        # rmsnorm initialized to 1
        # weight refers to the g_i learnable param
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Float[torch.Tensor, " ... d_model"]) -> Float[torch.Tensor, " ... d_model"]:
        """
        Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, d_model)
        """

        in_dtype = x.dtype
        # upcast before performing normalization to prevent overflow
        x = x.to(torch.float32)

        # RMSNorm(a_i) = (a_i / RMS(a)) * g_i
        # where RMS(a) = sqrt(mean(a^2) + eps)
        #             = sqrt((1/d_model) * sum(a_i^2) + eps)
        rms_a = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        rmsnorm = (x / rms_a) * self.weight

        # downcast back to original type
        return rmsnorm.to(in_dtype)
