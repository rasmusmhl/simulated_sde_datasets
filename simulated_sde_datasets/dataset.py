import jax
import jax.numpy as jnp
import diffrax
import distrax
from copy import deepcopy
from typing import Optional, Sequence

from .utils import dataloader


class Dataset:
    """ """

    def __init__(self, time_series_dataset: bool = True):
        self.data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.time_series_dataset = time_series_dataset

    def __getitem__(self, index):
        assert self.data is not None, "Data has not been setup."
        return self.data[index]

    def __len__(self):
        return self.data.shape[0]

    def make_partitions(self, train_pct=0.8, val_pct=0.1):
        dataset_size = len(self)
        n_train = int(dataset_size * train_pct)
        n_val = int(dataset_size * val_pct)
        # TODO: These do not include random permutations; should they?
        self.train_data = self.data[:n_train]
        self.val_data = self.data[n_train : n_train + n_val]
        self.test_data = self.data[n_train + n_val :]

    def get_dataloaders(self, key, batch_size):
        train_key, val_key, test_key, key = jax.random.split(key, 4)

        if self.data is not None and self.train_data is None:
            print("Assuming partitions should be made from self.data.")
            self.make_partitions()

        dl_train, dl_val, dl_test = None, None, None

        def make_dict(x):
            out = {"data": x}
            if self.time_series_dataset:
                n = x[0].shape[0]
                out["ts"] = jnp.tile(self.ts.reshape(1, -1), [n, 1])
            return out

        if self.train_data is not None:
            dl_train = dataloader(
                make_dict(self.train_data),
                batch_size,
                loop=True,
                key=train_key,
            )

        if self.val_data is not None:
            n_val = self.val_data[0].shape[0]
            dl_val = dataloader(
                make_dict(self.val_data),
                batch_size if batch_size < n_val else n_val,
                loop=True,
                key=val_key,
            )

        if self.test_data is not None:
            n_test = self.test_data[0].shape[0]
            dl_test = dataloader(
                make_dict(self.test_data),
                batch_size if batch_size < n_test else n_test,
                loop=False,
                key=test_key,
            )

        return dl_train, dl_val, dl_test


class SimulatedSDEDatateset(Dataset):
    def __init__(
        self,
        dataset_size: int,
        data_dimensions: int,
        brownian_motion_dimensions: int,
        brownian_motion_scale: float,
        t0: float = 0.0,
        t1: float = 1.0,
        dt0: float = 0.025,
        solver=diffrax.Midpoint(),
        burn: int = 0,
        brownian_motion_tolerance: float = 1e-3,
        normalize: bool = True,
        weakly_diagonal: bool = True,
        observation_model: distrax.Distribution = distrax.Normal(loc=0.0, scale=0.01),
        store_brownian_motion_keys: bool = False,
        hold_out_tails: bool = False,
        tail_quantile: Optional[float] = None,
    ):
        super().__init__()
        self.data_dimensions = data_dimensions
        self.dataset_size = dataset_size
        self.normalize = normalize
        self.observation_model = observation_model

        self.t0 = t0
        self.t1 = t1
        self.dt0 = dt0

        self.weakly_diagonal = weakly_diagonal
        if not weakly_diagonal:
            print("Not WeaklyDiagonal; are we sure?")

        # TODO: should we do different trajectory lengths both here and during training?
        self.ts = jnp.arange(self.t0, self.t1, self.dt0)

        self.solver = solver

        # TODO: is there a notion of momentum in this,
        # that we should be aware of in the data?
        # We remove an initial transient? like this:
        self.burn = burn
        # NOTE: this would be an issue if we had a dependence
        # on t (which we dont) in the (learnt) system, if we
        # didn't also change t0/t1

        self.brownian_motion_tolerance = brownian_motion_tolerance
        self.brownian_motion_dimensions = brownian_motion_dimensions
        self.brownian_motion_scale = brownian_motion_scale
        self._brownian_motion_scale = brownian_motion_scale * jnp.ones(
            [
                self.brownian_motion_dimensions,
            ]
        )
        self.args = None

        self.store_brownian_motion_keys = store_brownian_motion_keys
        self.brownian_motion_keys = None

        self.hold_out_tails = hold_out_tails
        self.tail_quantile = tail_quantile
        self.data_tails = None
        self.data_all = None

    def solve(self, y0, bm_key, args=None):
        brownian_motion = diffrax.VirtualBrownianTree(
            self.t0,
            self.t1,
            tol=self.brownian_motion_tolerance,
            shape=(self.brownian_motion_dimensions,),
            key=bm_key,
        )
        if self.weakly_diagonal:
            ct = diffrax.WeaklyDiagonalControlTerm(self.diffusion, brownian_motion)
        else:
            ct = diffrax.ControlTerm(self.diffusion, brownian_motion)
        terms = diffrax.MultiTerm(
            diffrax.ODETerm(self.drift),
            ct,
        )
        sol = diffrax.diffeqsolve(
            terms,
            self.solver,
            self.t0,
            self.t1,
            self.dt0,
            y0,
            saveat=diffrax.SaveAt(ts=self.ts),
            # adjoint=diffrax.NoAdjoint(),
            args=args,
        )
        return sol.ys

    def get_trajectories(self, key):
        initial_state_key, key = jax.random.split(key, 2)
        brownian_motion_keys = jax.random.split(key, self.dataset_size)
        if self.store_brownian_motion_keys:
            self.brownian_motion_keys = brownian_motion_keys
        y0 = self.get_initial_conditions(initial_state_key)
        ys = jax.vmap(self.solve)(
            y0,
            brownian_motion_keys,
            self.args,
        )
        return ys

    def get_held_out_tail(self, quantile=0.9, batch_size=16, key=jax.random.key(0)):
        # if quantile is a single float, we get everything above
        # but if its a list of floats, we get everything in the interval
        max_abs = jnp.max(jnp.abs(self.data_all), axis=[1, 2])
        if isinstance(quantile, Sequence):
            assert len(quantile) == 2
            limleft = jnp.quantile(max_abs, quantile[0])
            limright = jnp.quantile(max_abs, quantile[1])
            mask = (max_abs > limleft) & (max_abs < limright)
        else:
            mask = max_abs > jnp.quantile(max_abs, quantile)

        def make_dict(x):
            out = {"data": x}
            if self.time_series_dataset:
                n = x[0].shape[0]
                out["ts"] = jnp.tile(self.ts.reshape(1, -1), [n, 1])
            return out

        data = self.data_all[mask]
        return dataloader(
            make_dict(data),
            batch_size=data.shape[0],
            loop=False,
            key=key,
        )

    def partition_tails(self):
        max_abs = jnp.max(jnp.abs(self.data), axis=[1, 2])
        mask = max_abs < jnp.quantile(max_abs, self.tail_quantile)

        self.data_all = deepcopy(self.data)
        self.data = deepcopy(self.data[mask])
        self.data_tails = deepcopy(self.data_all[~mask])

    def setup(self, key):
        trajectory_key, noise_key = jax.random.split(key, 2)
        trajectories = self.get_trajectories(trajectory_key)

        if self.normalize:
            self.data_mu = trajectories.reshape(-1, self.data_dimensions).mean(0)
            self.data_sigma = trajectories.reshape(-1, self.data_dimensions).std(0)

            ys_noise_free = (trajectories - self.data_mu) / self.data_sigma
        else:
            ys_noise_free = trajectories

        noise = self.observation_model.sample(
            seed=noise_key,
            sample_shape=ys_noise_free.shape,
        )
        ys = ys_noise_free + noise

        self.data = ys[:, self.burn :, :]
        self.ts = self.ts[self.burn :]
        self.t0, self.t1 = self.ts[0], self.ts[-1]

        if self.hold_out_tails:
            print("Holding out tails.")
            self.partition_tails()
