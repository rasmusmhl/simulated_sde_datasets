import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt

from .dataset import SimulatedSDEDatateset
from .diffusion_data import student_diffusion


class StochasticLorenzAttractorDataset(SimulatedSDEDatateset):
    """
    Generate a dataset corresponding to stochastic system described in [1, Sec. 9.10.2].
    Defaults match description in their appendix.

    TODO: some of these values don't match the example on torchsde (see [2])?

    [1] https://arxiv.org/pdf/2001.01328.pdf
    [2] https://github.com/google-research/torchsde/blob/master/examples/latent_sde_lorenz.py
    """

    def __init__(
        self,
        sigma: float = 10,
        rho: float = 28,
        beta: float = 8 / 3,
        initial_conditions_scale: float = 1.0,
        dataset_size: int = 1024,
        burn: int = 0,
        brownian_motion_scale: float = 0.15,
        brownian_motion_dimensions: int = 3,
        **kwargs,
    ):
        super().__init__(
            data_dimensions=3,
            dataset_size=dataset_size,
            burn=burn,
            brownian_motion_scale=brownian_motion_scale,
            brownian_motion_dimensions=brownian_motion_dimensions,
            **kwargs,
        )

        self.sigma = sigma
        self.rho = rho
        self.beta = beta

        self.initial_conditions_scale = initial_conditions_scale

    def drift(self, t, y, args=None):
        y1, y2, y3 = y
        return jnp.array(
            [
                self.sigma * (y2 - y1),
                y1 * (self.rho - y3) - y2,
                y1 * y2 - self.beta * y3,
            ]
        )

    def diffusion(self, t, y, args=None):
        return self._brownian_motion_scale

    def get_initial_conditions(self, key):
        y0 = self.initial_conditions_scale * random.normal(
            key,
            shape=(
                self.dataset_size,
                self.data_dimensions,
            ),
        )
        return y0

    def visualize_data(self):
        projection = "3d"
        for traj in range(64):
            ax = plt.subplot(1, 1, 1, projection=projection)
            cur = self[traj] * self.data_sigma + self.data_mu
            plt.plot(
                *[cur[:, dim] for dim in range(self.data_dimensions)], "b", alpha=0.5
            )

        ax.set_xlabel("$X_1$")
        ax.set_ylabel("$X_2$")
        ax.set_zlabel("$X_3$")


class StudentStochasticLorenzAttractorDataset(StochasticLorenzAttractorDataset):
    def __init__(
        self,
        student_specs={
            "theta": 1.0,
            "mu": 0.0,
            "delta": 1.0,
            "nu": 2,
        },
        **kwargs,
    ):
        self.student_specs = student_specs
        super().__init__(**kwargs)

    def diffusion(self, t, y, args=None):
        return student_diffusion(t, y, **self.student_specs)
