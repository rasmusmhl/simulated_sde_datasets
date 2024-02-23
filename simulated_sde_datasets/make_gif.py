import jax
import jax.numpy as jnp
import distrax as dsx
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from . import SquareForcingStudentDiffusionDataset, SquareForcingOUDiffusionDataset

sns.set(style="whitegrid")

dataset_specs = {
    "dataset_size": 4,
    "dt0": 0.01,
    "t1": 5,
    "observation_model": dsx.Normal(0.0, 1e-2),
}
sq_st_dataset = SquareForcingStudentDiffusionDataset(
    **dataset_specs, store_brownian_motion_keys=True
)
sq_ou_dataset = SquareForcingOUDiffusionDataset(
    **dataset_specs, store_brownian_motion_keys=True
)
data_generation_key = jax.random.PRNGKey(21)
sq_st_dataset.setup(key=data_generation_key)
sq_ou_dataset.setup(key=data_generation_key)

fig, ax = plt.subplots(2, 2, figsize=(10, 10))
idx = 0
debug = False

lim = 0.0
for dataset in [sq_st_dataset, sq_ou_dataset]:
    for idx in range(4):
        lim_ = jnp.abs(dataset[idx]).max()
        if lim_ > lim:
            lim = lim_
        n_frames = 100 if not debug else 2
        n_time_steps = len(dataset.ts)
        T = dataset.ts.max()
        duration = 5 if not debug else 1
        fps = n_frames / duration
lim *= 1.1

actual_n_frames = n_frames + 1


def animate(frame_count):
    i = int((frame_count) / n_frames * n_time_steps)

    lines = ()
    idx = 0
    for r in range(2):
        for c in range(2):
            ax[r, c].clear()
            ax[r, c].set_xlim(0, T)
            ax[r, c].set_ylim(-lim, lim)
            names = ["Light-tailed", "Heavy-tailed"]
            datasets = [sq_ou_dataset, sq_st_dataset]
            for name, dataset in zip(names, datasets):
                for full in [True, False]:
                    alpha = 0.2 if full else 1.0
                    alpha_forcings = 0.1 if full else 0.2
                    ts = dataset.ts if full else dataset.ts[:i]
                    ys = dataset[idx, :, 0] if full else dataset[idx, :i, 0]
                    color = {"Light-tailed": "b", "Heavy-tailed": "orange"}[name]
                    lines += tuple(
                        ax[r, c].plot(
                            ts,
                            ys,
                            label=name if not full else None,
                            alpha=alpha,
                            color=color,
                        )
                    )
                    ax[r, c].plot(dataset.ts[i], dataset[idx, i, 0], "o", color=color)

                    if dataset.args is not None:
                        args = {k: v[idx, 0] for k, v in dataset.args.items()}
                        f = dataset.forcing_function(ts, None, args)
                        ax[r, c].fill_between(
                            ts,
                            f,
                            color="r",
                            alpha=alpha_forcings,
                            label=(
                                "Forcing"
                                if name == "Heavy-tailed" and not full
                                else None
                            ),
                        )

                    if (
                        name == "Heavy-tailed"
                        and dataset.brownian_motion_keys is not None
                    ):
                        bm = dfx.VirtualBrownianTree(
                            dataset.t0,
                            dataset.t1,
                            tol=dataset.brownian_motion_tolerance,
                            shape=(dataset.brownian_motion_dimensions,),
                            key=dataset.brownian_motion_keys[idx],
                        )

                        ax[r, c].fill_between(
                            ts,
                            jax.vmap(bm.evaluate)(ts).flatten(),
                            color="g",
                            alpha=alpha_forcings,
                            label="Brownian motion" if not full else None,
                        )

                    ax[r, c].set_yscale("symlog")
            if idx == 0:
                ax[r, c].legend(
                    loc="upper center",
                    bbox_to_anchor=(1.1, 1.2),
                    fancybox=True,
                    shadow=True,
                    ncol=4,
                )
            idx += 1
            if r == 0:
                ax[r, c].set_xticklabels([])
            else:
                ax[r, c].set_xlabel("Time")
            if c == 1:
                ax[r, c].set_yticklabels([])
            else:
                ax[r, c].set_ylabel("State")
    return lines


ani = FuncAnimation(
    fig,
    animate,
    interval=40,
    blit=True,
    repeat=True,
    frames=actual_n_frames,
)
ani.save("notebooks/forced_diffusion.gif", dpi=300, writer=PillowWriter(fps=fps))
plt.savefig("notebooks/forced_diffusion.png", dpi=300)
