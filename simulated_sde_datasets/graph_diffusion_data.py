import jraph
import jax.numpy as jnp
import matplotlib.pyplot as plt

from .graph_utils import build_circular_graph, draw_jraph_graph_structure
from .diffusion_data import StudentDiffusionDataset


class GraphStudentDiffusionDataset(StudentDiffusionDataset):
    def __init__(self, **kwargs):
        self.graph = build_circular_graph(N=48, max_meshes=None)
        self.graph_op = jraph.GraphConvolution(
            lambda x: x,
            aggregate_nodes_fn=jraph.segment_mean,
            add_self_edges=True,
            symmetric_normalization=True,
        )
        assert "data_dimensions" not in kwargs

        super().__init__(data_dimensions=int(self.graph.n_node), **kwargs)

    def graph_state(self, y):
        graph = self.graph._replace(nodes=y.reshape(-1, 1))
        out = self.graph_op(graph).nodes.reshape(-1)
        return out

    def drift(self, t, y, args):
        return super().drift(t, self.graph_state(y), args)

    def diffusion(self, t, y, args):
        return super().diffusion(t, self.graph_state(y), args)

    def visualize_data(self, as_image=True):
        plt.figure(figsize=(5, 5))
        draw_jraph_graph_structure(self.graph)
        spacer = 10
        plt.figure(figsize=(10, 10))
        for i in range(9):
            x = self.data[i]
            plt.subplot(3, 3, 1 + i)
            if as_image:
                x = jnp.log(jnp.abs(x))
                x = jnp.clip(x, a_min=-3)
                plt.imshow(x.transpose(), aspect="auto", interpolation="none")
                title_str = "Log-absolute magnitudes"
                plt.colorbar()
            else:
                for d in range(x.shape[-1]):
                    offset = spacer * d
                    y2 = x[:, d] + offset
                    y1 = jnp.zeros_like(self.ts) + offset
                    plt.fill_between(self.ts, y1=y1, y2=y2, alpha=0.5)
                plt.yticks([])
                title_str = "Magnitudes"
            if i in [0, 3, 6]:
                plt.ylabel("Node index")
            else:
                plt.gca().set_yticklabels([])
            if i in [6, 7, 8]:
                plt.xlabel("Time (a.u.)")
            else:
                plt.gca().set_xticklabels([])

        plt.suptitle(title_str)
        plt.show()
