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

# Step 8 - init_velocity_mlp
def init_velocity_mlp(in_dim, hidden_dim, time_embed_dim, seed):
    """Return a seeded parameter dict for a two-hidden-layer velocity MLP."""
    # TODO: Return a seeded parameter dict for a two-hidden-layer velocity MLP....
    torch.manual_seed(seed)

    #layer 1
    n_in=in_dim + time_embed_dim
    n_out=hidden_dim
    W1= torch.randn(n_in,n_out)*torch.sqrt(torch.tensor(2/(n_in+n_out))).requires_grad_()
    b1=torch.zeros(n_out, requires_grad=True)

    #layer 2
    W2=torch.randn(hidden_dim,hidden_dim)*torch.sqrt(torch.tensor(2/(hidden_dim+hidden_dim))).requires_grad_()
    b2=torch.zeros(hidden_dim, requires_grad=True)

    #layer 3
    W3=torch.randn(hidden_dim,in_dim)*torch.sqrt(torch.tensor(2/(hidden_dim+in_dim))).requires_grad_()
    b3=torch.zeros(in_dim, requires_grad=True)

    return {"W1":W1, "W2":W2,"W3":W3,"b1":b1,"b2":b2,"b3":b3}

# Step 9 - velocity_mlp_forward
def velocity_mlp_forward(x, t, params, time_embed_dim):
    """Return a velocity of the same shape as x from states, times, and MLP params."""
    # TODO: Embed t with the sinusoidal embedding, concatenate it with x...
    
    t = torch.as_tensor(t,dtype=x.dtype,device=x.device)
    t=t.reshape(-1,1).expand(x.shape[0],1)
    emb = sinusoidal_time_embedding(t,time_embed_dim)
    h=torch.cat([x,emb],dim=1)
    h=torch.relu(h@params["W1"]+params["b1"])
    h=torch.relu(h@params["W2"]+params["b2"])
    v=h@params["W3"]+params["b3"]
    return v

# Step 10 - make_mixture_dataset
def make_mixture_dataset(centers, n_per_component, std, seed):
    """Build a fixed-seed 2-D Gaussian mixture from component centers."""
    # TODO: Build a fixed-seed 2-D Gaussian mixture from component centers...
    torch.manual_seed(seed)
    groups=[]
    # centers #shape n_components,2
    # n_per_component #int
    # std #float
    for k in range(centers.shape[0]): #how many clouds
        Z=torch.randn(n_per_component,2) #how many dots
        groups.append(centers[k]+Z*std)
    return torch.cat(groups,dim=0).to(torch.float32)

# Step 11 - flow_train_step
def flow_train_step(params, data, batch_size, lr, time_embed_dim, seed):
    """Draw a minibatch of data, compute the flow-matching loss through the velocity MLP, and take one manual SGD update."""
    # TODO: Implement flow_train_step to perform one in-place manual SGD update on a velocity MLP.
    torch.manual_seed(seed)
    # pick some rows
    n=data.shape[0]
    idx=torch.randint(0,n,(batch_size,))
    minibatch=data[idx]

    # make examples guess score
    xt, v_target, t = make_flow_batch(minibatch,seed+1)
    v_pred = velocity_mlp_forward(xt,t,params,time_embed_dim)
    loss = torch.nn.functional.mse_loss(v_target,v_pred)

    # find slope
    for p in params.values():
        if p.grad is not None:
            p.grad.zero_() #compute new slope
    loss.backward()

    # nudge each weight
    with torch.no_grad():
        for p in params.values():
            p -= lr * p.grad

    return params,loss.item()

