import jax
import jax.numpy as jnp
import jraph
import networkx as nx


def convert_jraph_to_networkx_graph(jraph_graph: jraph.GraphsTuple) -> nx.Graph:
    nodes, edges, receivers, senders, _, _, _ = jraph_graph
    nx_graph = nx.DiGraph()
    if nodes is None:
        for n in range(jraph_graph.n_node[0]):
            nx_graph.add_node(n)
    else:
        for n in range(jraph_graph.n_node[0]):
            nx_graph.add_node(n, node_feature=nodes[n])
    if edges is None:
        for e in range(jraph_graph.n_edge[0]):
            nx_graph.add_edge(int(senders[e]), int(receivers[e]))
    else:
        for e in range(jraph_graph.n_edge[0]):
            nx_graph.add_edge(int(senders[e]), int(receivers[e]), edge_feature=edges[e])
    return nx_graph


def draw_jraph_graph_structure(jraph_graph: jraph.GraphsTuple) -> None:
    nx_graph = convert_jraph_to_networkx_graph(jraph_graph)
    # pos = nx.spring_layout(nx_graph)
    pos = nx.shell_layout(nx_graph)
    # pos = nx.spectral_layout(nx_graph)
    # pos = nx.kamada_kawai_layout(nx_graph)
    nx.draw(nx_graph, pos=pos, with_labels=True, node_size=500, font_color="yellow")


def build_circular_graph(N=48, max_meshes=None) -> jraph.GraphsTuple:
    """Construct a graph with a multimesh structure."""
    # Doesn't work all N right now
    # Make base graph/circle
    s_base = jnp.arange(0, N)
    r = jnp.roll(s_base, shift=1)
    s = s_base

    # Add consectutively coarser meshes
    n_meshes = jnp.floor(jnp.log2(N))
    cur_s = s = s_base
    if max_meshes is not None:
        n_meshes = jnp.clip(n_meshes, a_max=max_meshes)
    n_meshes = int(n_meshes)
    for _ in range(n_meshes):
        ms = cur_s[::2]
        mr = jnp.roll(ms, shift=-1)

        s = jnp.concatenate([s, ms])
        r = jnp.concatenate([r, mr])
        cur_s = ms

    # Add opposite direction for each edge
    senders = jnp.concatenate([s, r])
    receivers = jnp.concatenate([r, s])

    edges = None
    node_features = jnp.ones([N, 1])

    n_node = jnp.array([N])
    n_edge = jnp.array([senders.shape[0]])

    global_context = jnp.array([[1]])

    graph = jraph.GraphsTuple(
        nodes=node_features,
        edges=edges,
        senders=senders,
        receivers=receivers,
        n_node=n_node,
        n_edge=n_edge,
        globals=global_context,
    )
    return graph
