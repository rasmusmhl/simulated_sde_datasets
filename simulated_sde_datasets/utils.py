import jax
import jax.numpy as jnp


def dataloader(dataset_dict, batch_size, loop, key, dataset_size=None):
    dataset_size = (
        dataset_dict["data"].shape[0] if dataset_size is None else dataset_size
    )

    indices = jnp.arange(dataset_size)

    assert batch_size <= dataset_size, "Too large batchsize"
    while True:
        permutation_key, key = jax.random.split(key, 2)
        perm = jax.random.permutation(permutation_key, indices)
        start = 0
        end = batch_size
        while end <= dataset_size:
            batch_perm = perm[start:end]
            yield jax.tree_map(lambda x: x[batch_perm, ...], dataset_dict)
            start = end
            end = start + batch_size
        if not loop:
            break
