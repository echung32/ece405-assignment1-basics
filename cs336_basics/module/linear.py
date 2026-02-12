import einx
import torch
import torch.nn as nn
import math


class Linear(nn.Module):
    """
    Linear transformation module: y = Wx
    follow the interface of pytorch nn.Linear module except for not having bias arg/parma (which most modern LLMs don't)
    """
    
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        """
        Construct a linear transformation module.
        
        Args:
            in_features: int - final dimension of the input
            out_features: int - final dimension of the output
            device: torch.device | None - Device to store the parameters on
            dtype: torch.dtype | None - Data type of the parameters
        """
        super().__init__()

        # d_in
        self.in_features = in_features
        # d_out
        self.out_features = out_features
        
        # "construct and store your parameter as W (not W ⊤) for memory ordering reasons, putting it in nn.Parameter"
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        
        # initialize weights using truncated normal distribution
        # σ² = 2 / (d_in + d_out)
        std = math.sqrt(2.0 / (in_features + out_features))
        # Truncate at [-3σ, 3σ]
        with torch.no_grad():
            nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the linear transformation to the input.
        
        Args:
            x: torch.Tensor - input tensor with shape (..., in_features)
            
        Returns:
            torch.Tensor - output tensor with shape (..., out_features)
        """
        # perform linear transformation: y = Wx
        # print(x.shape, self.weight.shape, self.weight.t().shape, torch.matmul(x, self.weight.t()).shape)
        # return torch.matmul(x, self.weight.t())

        # torch.testing.assert_close(einx.dot("... in, in out -> ... out", x, self.weight.t()), torch.matmul(x, self.weight.t()))

        # try using einx
        return einx.dot("... in, in out -> ... out", x, self.weight.t())

if __name__ == "__main__":
    d_in, d_out = 16, 32
    m = Linear(d_in, d_out)
    x = torch.randn(d_in)  # shape: (d_in,)
    y = m(x)
    assert y.shape == (d_out,)
    # 2D batch
    x = torch.randn(4, d_in)  # shape: (batch, d_in)
    y = m(x)
    assert y.shape == (4, d_out)

    # torch.Size([16]) torch.Size([32, 16]) torch.Size([16, 32]) torch.Size([32])
    # torch.Size([4, 16]) torch.Size([32, 16]) torch.Size([16, 32]) torch.Size([4, 32])
