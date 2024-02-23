import jax

from stochastic_lorenz_data import StochasticLorenzAttractorDataset


def test_stochastic_lorenz_data():
    dataset_size = 8
    dataset = StochasticLorenzAttractorDataset(dataset_size=dataset_size)
    dataset.setup(key=jax.random.PRNGKey(0))

    assert len(dataset)
    assert dataset[0].shape == (len(dataset.ts), dataset.data_dimensions)
