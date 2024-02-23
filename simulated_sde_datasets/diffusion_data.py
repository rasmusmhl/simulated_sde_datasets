from typing import Tuple
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from .dataset import SimulatedSDEDatateset


class BaseDiffusionDataset(SimulatedSDEDatateset):
    def visualize_data(self):
        plt.figure(figsize=(10, 10))
        for i in range(4):
            cur = self[i]
            plt.subplot(2, 2, i + 1)
            plt.plot(self.ts, cur[:, 0], "b", label="input data")
            plt.xlabel("Time")
            plt.ylabel("Magnitude")
            if self.args is not None:
                args = {k: v[i, 0] for k, v in self.args.items()}
                f = self.forcing_function(self.ts, None, args)
                plt.plot(self.ts, f, "r", label="forcing")

            plt.legend()

        plt.show()

        all_data_flat = self[: self.dataset_size].flatten()
        log_magnitudes = jnp.log(jnp.abs(all_data_flat))
        clip_log_magnitudes = jnp.clip(
            log_magnitudes,
            a_min=jnp.percentile(log_magnitudes, 1),
            a_max=None,
        )

        plt.figure(figsize=(10, 8))
        plt.subplot(2, 2, 1)
        plt.hist(all_data_flat, bins=100, density=True)
        plt.title("Histogram of data")
        plt.xlabel("Magnitude")
        plt.ylabel("Frequency")
        plt.subplot(2, 2, 2)

        plt.title("Log-log-histogram of data\n(clipped at 1st percentile)")
        plt.hist(clip_log_magnitudes, bins=100, density=True)
        plt.gca().set_yscale("log")
        plt.xlabel("Log-magnitude")
        plt.ylabel("Log-frequency")
        plt.show()


class OrnsteinUhlenbeckDiffusionDataset(BaseDiffusionDataset):
    """
    Implements an Ornstein-Uhlenbeck process [1].

    [1] https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
    """

    def __init__(
        self,
        dataset_size: int = 1024,
        mu: float = 0.0,
        theta=1.0,
        sigma=1.0,
        brownian_motion_scale: float = 1e-2,
        initial_conditions_limits: Tuple[int] = [-1, 1],
        data_dimensions: int = 1,
        **kwargs,
    ):
        super().__init__(
            data_dimensions=data_dimensions,
            brownian_motion_dimensions=data_dimensions,
            dataset_size=dataset_size,
            brownian_motion_scale=brownian_motion_scale,
            normalize=False,
            **kwargs,
        )
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.initial_conditions_limits = initial_conditions_limits

    def drift(self, t, y, args):
        return -self.theta * (y - self.mu)

    def diffusion(self, t, y, args):
        return self.sigma * jnp.ones_like(y)

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


def student_diffusion(t, y, theta, mu, delta, nu):
    coef = 2 * theta * (delta**2) / (nu - 1)
    scaled_loc = (y - mu) / delta
    val = 1 + scaled_loc**2
    return jnp.sqrt(coef * val)


class StudentDiffusionDataset(BaseDiffusionDataset):
    """
    Implements a heavy-tailed generalization of an OU process
    which we will call a Student(-T) Diffusion  process [1].

    For t >= 0:

    d Xt = b(Xt, t) dt + sigma(Xt, t) d Wt

    where the drift, b is:

    b(Xt, t) = - theta (Xt - mu )

    and the diffusion sigma is:

    sigma(Xt, t) = sqrt( c * f(Xt,t) ),
    c = (2 * theta * delta ** 2 ) / ( nu - 1 )
    f(Xt, t) = 1 + ((Xt - mu)/delta)**2

    and Wt is a brownian motion.

    [1] https://www.tandfonline.com/doi/full/10.1080/07362994.2010.515476
    """

    def __init__(
        self,
        dataset_size: int = 1024,
        theta=1.0,
        mu=0.0,
        delta=1.0,
        nu=2,
        brownian_motion_scale: float = 1e-2,
        initial_conditions_limits: Tuple[int] = [-1, 1],
        data_dimensions: int = 1,
        **kwargs,
    ):
        super().__init__(
            data_dimensions=data_dimensions,
            brownian_motion_dimensions=data_dimensions,
            dataset_size=dataset_size,
            brownian_motion_scale=brownian_motion_scale,
            normalize=False,
            weakly_diagonal=True,
            **kwargs,
        )

        self.theta = theta
        self.mu = mu
        self.delta = delta
        self.nu = nu
        self.initial_conditions_limits = initial_conditions_limits

    def drift(self, t, y, args):
        return -self.theta * (y - self.mu)

    def diffusion(self, t, y, args):
        return student_diffusion(t, y, self.theta, self.mu, self.delta, self.nu)

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
