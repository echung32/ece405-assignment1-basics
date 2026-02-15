from cs336_basics.optimizer.sgd import SGD
import torch


def test_sgd_with_learning_rate(lr: float, num_iters: int = 10):
    """
    Test SGD optimizer with a given learning rate.
    
    Args:
        lr: Learning rate to test
        num_iters: Number of training iterations
    """
    torch.manual_seed(42)
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=lr)
    
    print(f'\n=== Testing SGD with lr={lr} ===')
    losses = []
    
    for t in range(num_iters):
        opt.zero_grad()
        loss = (weights**2).mean()
        losses.append(loss.cpu().item())
        print(f'Iteration {t}: loss = {loss.cpu().item():.6e}')
        loss.backward()
        opt.step()
    
    return losses


if __name__ == "__main__":
    print("=" * 80)
    print("SGD Optimizer Learning Rate Tuning Experiment")
    print("=" * 80)
    
    # Test with the original learning rate from the example
    print("\n1. Baseline test with lr=1")
    test_sgd_with_learning_rate(lr=1, num_iters=10)
    
    # Test with the requested learning rates
    print("\n2. Testing with different learning rates (10 iterations each)")
    learning_rates = [1e1, 1e2, 1e3]
    
    for lr in learning_rates:
        losses = test_sgd_with_learning_rate(lr=lr, num_iters=10)
        
        # Analyze behavior
        if losses[-1] < losses[0]:
            behavior = "decays (converging)"
        elif losses[-1] > losses[0] * 10:
            behavior = "diverges (exploding)"
        else:
            behavior = "stable or slowly changing"
        
        print(f"Behavior: {behavior}")
