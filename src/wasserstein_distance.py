import torch
import torch.nn as nn


def wasserstein_1_distance(predicted, targets):
    """
    Compute Wasserstein-1 distance between two 1x3 tensors using the specified subroutine.
    All operations are differentiable using PyTorch.
    
    Args:
        predicted (torch.Tensor): 1x3 tensor of predicted values
        targets (torch.Tensor): 1x3 tensor of target values
    
    Returns:
        torch.Tensor: Wasserstein-1 distance (scalar tensor)
    """
    # Ensure inputs are torch tensors
    if not isinstance(predicted, torch.Tensor):
        predicted = torch.tensor(predicted, dtype=torch.float32)
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets, dtype=torch.float32)
    
    # Step 1: Rearrange the data
    # Form c_predicted as [predicted[-1], 1-predicted[-1]]
    c_predicted = torch.stack([predicted[-1], 1 - predicted[-1]])
    
    # Form c_target as [targets[-1], 1-targets[-1]]
    c_target = torch.stack([targets[-1], 1 - targets[-1]])
    # Step 2: Form T2_full as concatenation of predicted[:-1] and targets[:-1]
    T2_full = torch.cat([predicted[:-1], targets[:-1]])
    
    # Form c_predicted_full as concatenation of c_predicted with zeros of same length
    c_predicted_full = torch.cat([c_predicted, torch.zeros_like(c_predicted)])
    
    # Form c_target_full as zeros followed by c_target
    c_target_full = torch.cat([torch.zeros_like(c_target), c_target])
    
    # Step 3: Sort T2_full in descending order and rearrange c arrays in same order
    # Get both sorted values and indices in descending order
    T2_full_sorted, sort_indices = torch.sort(T2_full, descending=True)
    # Apply the sorting to the c arrays using the indices
    c_predicted_full_sorted = c_predicted_full[sort_indices]
    c_target_full_sorted = c_target_full[sort_indices]
    
    # Step 4: Form cumulative sums and absolute sequential differences
    # Cumulative sums
    cum_c_predicted = torch.cumsum(c_predicted_full_sorted, dim=0)
    cum_c_target = torch.cumsum(c_target_full_sorted, dim=0)
    # Take reciprocals of sorted T2 and append zero to avoid infinities
    T2_reciprocals = 1.0 / T2_full_sorted
    # T2_reciprocals_with_zero = torch.cat([T2_reciprocals, torch.zeros(1, dtype=T2_reciprocals.dtype, device=T2_reciprocals.device)])
    
    # Absolute sequential differences of the reciprocals
    sequential_diffs = torch.abs(torch.diff(T2_reciprocals))
    sequential_diffs = torch.cat([sequential_diffs, torch.zeros(1, dtype=sequential_diffs.dtype, device=sequential_diffs.device)]) 
    # Step 5: Return the dot product of absolute differences between cumulative sums 
    # and the absolute sequential differences
    abs_cum_diff = torch.abs(cum_c_predicted - cum_c_target)
    wasserstein_distance = torch.dot(abs_cum_diff, sequential_diffs)
    return wasserstein_distance

def wasserstein_1_distance_batched(predicted: torch.Tensor, targets: torch.Tensor):
    """
    Compute Wasserstein-1 distance between batched tensors using the specified subroutine.
    All operations are differentiable using PyTorch.
    
    Args:
        predicted (torch.Tensor): batch_size x 3 tensor of predicted values
        targets (torch.Tensor): batch_size x 3 tensor of target values
    
    Returns:
        torch.Tensor: Wasserstein-1 distances for each sample in batch (batch_size tensor)
    """
    
    # Step 1: Rearrange the data
    # Form c_predicted as [predicted[-1], 1-predicted[-1]]
    c_predicted = torch.stack([predicted[..., -1], 1 - predicted[..., -1]], dim=-1)
    # Form c_target as [targets[-1], 1-targets[-1]]
    c_target = torch.stack([targets[..., -1], 1 - targets[..., -1]], dim=-1)

    # Step 2: Form T2_full as concatenation of predicted[:-1] and targets[:-1]
    T2_full = torch.cat([predicted[..., :-1], targets[..., :-1]], dim=-1)
    # Form c_predicted_full as concatenation of c_predicted with zeros of same length
    c_predicted_full = torch.cat([c_predicted, torch.zeros_like(c_predicted)], dim=-1)
    # Form c_target_full as zeros followed by c_target
    c_target_full = torch.cat([torch.zeros_like(c_target), c_target], dim=-1)

    # Step 3: Sort T2_full in descending order and rearrange c arrays in same order
    # Get both sorted values and indices in descending order
    T2_full_sorted, sort_indices = torch.sort(T2_full, descending=True, dim=-1, stable=True)
    # Apply the sorting to the c arrays using the indices
    c_predicted_full_sorted = torch.gather(c_predicted_full, dim=-1, index=sort_indices)
    c_target_full_sorted = torch.gather(c_target_full, dim=-1, index=sort_indices)

    # Step 4: Form cumulative sums and absolute sequential differences
    # Cumulative sums
    cum_c_predicted = torch.cumsum(c_predicted_full_sorted, dim=-1)
    cum_c_target = torch.cumsum(c_target_full_sorted, dim=-1)
    # Take reciprocals of sorted T2 and append zero to avoid infinities
    T2_reciprocals = 1.0 / T2_full_sorted
    # T2_reciprocals_with_zero = torch.cat([T2_reciprocals, torch.zeros(1, dtype=T2_reciprocals.dtype, device=T2_reciprocals.device)])
    # Absolute sequential differences of the reciprocals
    sequential_diffs = torch.abs(torch.diff(T2_reciprocals))

    # Step 5: Return the dot product of absolute differences between cumulative sums 
    # and the absolute sequential differences
    abs_cum_diff = torch.abs(cum_c_predicted - cum_c_target)
    dot_prods = torch.sum(abs_cum_diff[:, :-1] * sequential_diffs, dim=-1) # only all but the last element of the cumulative sums are needed for the integral
    return dot_prods


class WassersteinLoss(nn.Module):
    """
    PyTorch loss module for computing Wasserstein-1 distance over batched tensors.
    
    This loss function computes the Wasserstein-1 distance for each sample in a batch
    and applies the specified reduction.
    """
    
    def __init__(self, reduction='mean'):
        """
        Initialize the Wasserstein loss module.
        
        Args:
            reduction (str): Specifies the reduction to apply to the output:
                           'mean' | 'sum' | 'none'. Default: 'mean'
        """
        super(WassersteinLoss, self).__init__()
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Invalid reduction mode: {reduction}. Must be 'mean', 'sum', or 'none'")
        self.reduction = reduction
    
    def forward(self, predicted: torch.Tensor, targets: torch.Tensor):
        """
        Compute the Wasserstein-1 loss for batched tensors.
        
        Args:
            predicted (torch.Tensor): Predicted values of shape (batch_size, 3)
            targets (torch.Tensor): Target values of shape (batch_size, 3)
        
        Returns:
            torch.Tensor: Wasserstein-1 loss (scalar if reduction != 'none', 
                         otherwise tensor of shape (batch_size,))
        """
        # Ensure inputs are torch tensors
        if not isinstance(predicted, torch.Tensor):
            predicted = torch.tensor(predicted, dtype=torch.float32)
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets, dtype=torch.float32)
        
        # Validate input shapes
        if predicted.shape != targets.shape:
            raise ValueError(f"Shape mismatch: predicted {predicted.shape} vs targets {targets.shape}")
        if predicted.ndim != 2 or predicted.shape[-1] != 3:
            raise ValueError(f"Expected input shape (batch_size, 3), got {predicted.shape}")
        
        # Compute Wasserstein distances for the batch
        distances = wasserstein_1_distance_batched(predicted, targets)
        
        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(distances)
        elif self.reduction == 'sum':
            return torch.sum(distances)
        elif self.reduction == 'none':
            return distances
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


def example_usage():
    """
    Example usage of the Wasserstein-1 distance function.
    Demonstrates differentiability by computing gradients.
    """
    # Example tensors with gradient tracking
    # predicted = torch.tensor([0.3, 0.5, 0.8], dtype=torch.float32, requires_grad=True)
    # targets = torch.tensor([0.2, 0.4, 0.7], dtype=torch.float32)
    predicted = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=torch.float32, requires_grad=True)
    targets = torch.tensor([[1.0, 2.0, 0.2], [3.0, 4.0, 0.5], [6.0, 7.0, 0.7]], dtype=torch.float32)
    
    # Compute Wasserstein-1 distance using batched function
    distances = wasserstein_1_distance_batched(predicted, targets)
    mean_distance = torch.mean(distances)
    
    print(f"Predicted: {predicted}")
    print(f"Targets: {targets}")
    print(f"Individual Wasserstein-1 distances: {distances}")
    print(f"Mean Wasserstein-1 distance: {mean_distance}")
    
    # Demonstrate differentiability by computing gradients
    mean_distance.backward()
    print(f"Gradients w.r.t. predicted: {predicted.grad}")
    
    return mean_distance


def example_loss_module():
    """
    Example usage of the WassersteinLoss module.
    Demonstrates batched loss computation with different reduction modes.
    """
    # Example batched tensors with gradient tracking
    batch_size = 4
    predicted = torch.tensor([
        [0.3, 0.5, 0.8],
        [0.1, 0.7, 0.6],
        [0.4, 0.3, 0.9],
        [0.2, 0.6, 0.5]
    ], dtype=torch.float32, requires_grad=True)
    
    targets = torch.tensor([
        [0.2, 0.6, 0.7],
        [0.15, 0.65, 0.55],
        [0.35, 0.35, 0.85],
        [0.25, 0.55, 0.45]
    ], dtype=torch.float32)
    
    print("Wasserstein Loss Module Example:")
    print(f"Predicted shape: {predicted.shape}")
    print(f"Targets shape: {targets.shape}")
    
    # Test different reduction modes
    loss_fns = {
        'mean': WassersteinLoss(reduction='mean'),
        'sum': WassersteinLoss(reduction='sum'),
        'none': WassersteinLoss(reduction='none')
    }
    
    for reduction_name, loss_fn in loss_fns.items():
        # Clone tensors to avoid gradient issues
        pred_clone = predicted.clone().detach().requires_grad_(True)
        loss = loss_fn(pred_clone, targets)
        
        print(f"\nReduction '{reduction_name}': {loss}")
        if reduction_name != 'none':
            print(f"Loss shape: {loss.shape}")
        else:
            print(f"Individual losses shape: {loss.shape}")
        
        # Demonstrate backpropagation
        if reduction_name == 'mean':  # Only backward on mean to show gradients
            loss.backward()
            print(f"Gradients w.r.t. predicted:\n{pred_clone.grad}")
    
    return loss_fns['mean'](predicted, targets)


# # uncomment to run the examples
# if __name__ == "__main__":
#     print("Batched function example:")
#     example_usage()
#     print("\n" + "="*60 + "\n")
#     print("Loss module example:")
#     example_loss_module()
