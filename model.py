"""
Rectified Flow from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - interpolate_linear
def interpolate_linear(x0, x1, t):
    """Return the linear interpolant from noise at t=0 to data at t=1."""
    # TODO: Return the linear interpolant from noise at t=0 to data at t=1.
    return x0*(1-t) +x1*t

# Step 2 - target_velocity
def target_velocity(x0, x1):
    """Return the constant straight-path target velocity from noise x0 to data x1."""
    # TODO: Return the constant straight-path target velocity along the interpolant...
    return x1-x0

# Step 3 - sample_uniform_time
def sample_uniform_time(batch_size, seed):
    """Draw a column of times uniformly from the unit interval."""
    # TODO: Draw a column of times uniformly from the unit interval...
    torch.manual_seed(seed)

    #shape (batch_size, 1)
    return torch.rand(batch_size,1)

# Step 4 - sample_gaussian_noise
def sample_gaussian_noise(x, seed):
    """Return standard normal noise matching the shape of a given tensor using a given seed."""
    # TODO: Return standard normal noise matching a tensor using a given seed...
    torch.manual_seed(seed)
    return torch.randn_like(x)

# Step 5 - make_flow_batch
def make_flow_batch(x1, seed):
    """Given data x1, sample noise and time and return xt, target velocity, and t."""
    # TODO: Implement `make_flow_batch` to assemble one flow-matching training triple from a batch of data....
    batch_size=x1.shape[0]
    t=sample_uniform_time(batch_size,seed)
    x0=sample_gaussian_noise(x1,seed)
    xt=interpolate_linear(x0,x1,t)
    target=target_velocity(x0,x1)

    return (xt,target,t)

# Step 6 - flow_matching_loss
def flow_matching_loss(v_pred, v_target):
    """Score a predicted velocity against the straight-path target."""
    # TODO: Implement `flow_matching_loss` to score predicted velocities against a target.
    return torch.nn.functional.mse_loss(v_pred,v_target)

# Step 7 - sinusoidal_time_embedding
def sinusoidal_time_embedding(t, embed_dim):
    """Map each time in t to a sin-cos Fourier feature vector of length embed_dim."""
    # TODO: Map each time in t to a sin-cos Fourier feature vector of length embed_dim.
    half = embed_dim//2
    i = torch.arange(half , device = t.device, dtype=t.dtype)
    log10000=torch.log(torch.tensor(10000.0, device=t.device,dtype=t.dtype))
    freqs=torch.exp(-log10000*i/half)
    t=t.reshape(-1,1)
    angles=t*freqs
    return torch.cat([torch.sin(angles),torch.cos(angles)] , dim=-1)

    #half=8//2 = 4
    #sin sin sin sin cos cos cos cos

