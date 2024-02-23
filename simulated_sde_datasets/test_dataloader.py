import jax

from .stochastic_lorenz_data import StochasticLorenzAttractorDataset


def test_dataloader():
    batch_size = 2
    dataset = StochasticLorenzAttractorDataset(dataset_size=16)
    dataset.setup(key=jax.random.PRNGKey(0))
    dl, _, _ = dataset.get_dataloaders(jax.random.PRNGKey(0), batch_size)
    batch = next(dl)
    data, ts = batch["data"], batch["ts"]

    assert ts.shape[0] == batch_size
    assert data.shape[0] == batch_size
