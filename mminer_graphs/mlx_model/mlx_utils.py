import mlx.core as mx
import mlx.nn as mlx_nn
import mlx.optimizers as optim

MLX_OPTI = {
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "sgd": optim.SGD,
    "rmsprop": optim.RMSprop,
    "adagrad": optim.Adagrad,
    "adadelta": optim.AdaDelta,
    "adamax": optim.Adamax,
}

NO_WT_DECAY = ["adam", "rmsprop", "adagrad", "adadelta", "adamax"]

MLX_ACT = {
    "relu": mlx_nn.relu,
    "leaky_relu": lambda x: mlx_nn.leaky_relu(x, negative_slope=0.01),
    "gelu": mlx_nn.gelu,
    "elu": mlx_nn.elu,
    "silu": mlx_nn.silu,  # Also known as Swish
    "tanh": mlx_nn.tanh,
}

MLX_CRITERIONS = {
    # Classification
    "cross_entropy": mlx_nn.losses.cross_entropy,
    "nll": mlx_nn.losses.nll_loss,
    "kl_divergence": mlx_nn.losses.kl_div_loss,
    # "bce_with_logits": mlx_nn.losses.BCEWithLogitsLoss,
    # "poisson": mlx_nn.PoissonNLLLoss,
    # Regression
    "mse": mlx_nn.losses.mse_loss,
    "l1": mlx_nn.losses.l1_loss,
    "smooth_l1": mlx_nn.losses.smooth_l1_loss,
    "huber": mlx_nn.losses.huber_loss,
    # Others
    "cosine_similarity": mlx_nn.losses.cosine_similarity_loss,
    "triplet_margin": mlx_nn.losses.triplet_loss,
}


# support function to get apple silicon gpu, cpu, or cuda
def get_device(device_type="auto"):
    if device_type == "auto":
        device = mx.gpu
    elif device_type == "gpu":
        device = mx.gpu
    elif device_type == "cpu":
        device = mx.cpu
    elif device_type == "cuda":  # todo test on cuda enabled machine
        if not mx.cuda.is_available():
            raise ValueError("CUDA not available!")
        else:
            device = mx.cuda
    else:
        raise ValueError(
            f"Trying to run device on {device_type}. Only 'auto', 'cuda', 'gpu'(cuda), 'mps' or 'cpu' are supported!!")
    return device

# set seed for mlx
def set_seed(seed: int = 42) -> None:
    mx.random.seed(seed)

if __name__ == "__main__":
    import doctest

    doctest.testmod()
