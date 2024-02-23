import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from .dataset import SimulatedSDEDatateset
from .diffusion_data import student_diffusion


class StochasticLotkaVolterraDataset(SimulatedSDEDatateset):
    """
    We'll consider a stochastic version of the deterministic LV equations described
    in [1]:
        x' = alpha * x      - beta  * x * y
        y' = delta * x * y  - gamma * y
    In particular we'll follow the "baboon/cheetah" example.

    TODO: Generalize beyond the two-species setting, e.g. with [2]

    [1] https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations
    [2]https://arxiv.org/pdf/1703.04809.pdf
    """

    def __init__(
        self,
        dataset_size: int = 1024,
        alpha=1.1,
        beta=0.4,
        delta=0.1,
        gamma=0.4,
        brownian_motion_scale: float = 1e-2,
        initial_conditions_limits=[1, 10],
        **kwargs,
    ):
        super().__init__(
            data_dimensions=2,
            brownian_motion_dimensions=2,
            dataset_size=dataset_size,
            brownian_motion_scale=brownian_motion_scale,
            **kwargs,
        )

        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.gamma = gamma
        self.initial_conditions_limits = initial_conditions_limits

    def drift(self, t, y, args=None):
        y = jnp.clip(y, a_min=0)
        y1, y2 = y
        return jnp.array(
            [
                self.alpha * y1 - self.beta * y1 * y2,
                self.delta * y1 * y2 - self.gamma * y2,
            ]
        )

    def diffusion(self, t, y, args=None):
        return self._brownian_motion_scale

    def get_initial_conditions(self, key):
        y0 = jax.random.uniform(
            key,
            shape=(
                self.dataset_size,
                self.data_dimensions,
            ),
            minval=self.initial_conditions_limits[0],
            maxval=self.initial_conditions_limits[1],
        )
        return y0

    def visualize_data(self):
        for traj in range(64):
            cur = self[traj] * self.data_sigma + self.data_mu
            plt.plot(
                *[cur[:, dim] for dim in range(self.data_dimensions)],
                "b",
                alpha=0.5,
            )
        plt.xlabel("Prey")
        plt.ylabel("Predator")


class StudentStochasticLotkaVolterraDataset(StochasticLotkaVolterraDataset):

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
