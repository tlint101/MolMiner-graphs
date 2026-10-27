from .plot import Plots

#todo check if this statement works. if so, combine mlx_models with graph
# optional mlx import
try:
    from .mlx_plot import PlotsMLX
    __all__ = ["Plots", "PlotsMLX"]
except ImportError:
    __all__ = ["Plots"]
