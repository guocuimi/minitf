from minitf.autodiff.util import to_list
from minitf.tensor import is_tensor


class Graph(object):
    def __init__(self):
        self._edges = dict()

    def add_edges(self, source, targets, metadata=None):
        self._edges[source] = (to_list(targets), metadata)

    def get_targets(self, source, default=(None, None)):
        return self._edges.get(source, default)


__GRAPH_STACK = []


def get_current_graph():
    if __GRAPH_STACK:
        return __GRAPH_STACK[-1]
    return None


def push_graph(graph):
    __GRAPH_STACK.append(graph)


def pop_graph():
    return __GRAPH_STACK.pop()


def toposort(graph, source):
    child_counts = {}
    stack = [source]
    while stack:
        node = stack.pop()
        if node in child_counts:
            child_counts[node] += 1
        else:
            child_counts[node] = 1
            neighbors, _ = graph.get_targets(node, ([], []))
            stack.extend(neighbors)

    childless_nodes = [source]
    while childless_nodes:
        node = childless_nodes.pop()
        neighbors, vjps = graph.get_targets(node, ([], []))
        yield node, neighbors, vjps
        for neighbor in neighbors:
            if child_counts[neighbor] == 1:
                childless_nodes.append(neighbor)
            else:
                child_counts[neighbor] -= 1


def register_op(func, ans, *args, **kwargs):
    current_graph = get_current_graph()
    if current_graph:
        # make vjp functions
        from minitf.vjps.vjp_maker import get_vjp_maker
        vjp_maker = get_vjp_maker(func)
        if vjp_maker is None:
            raise Exception("Need to define vjp for the primitive")
        all_vjps = vjp_maker(ans, *args, **kwargs)

        tensors = []
        vjps = []
        for arg, vjp in zip(args, all_vjps):
            if is_tensor(arg):
                tensors.append(arg)
                vjps.append(vjp)

        # register grad func for each tensor
        current_graph.add_edges(ans, tensors, vjps)
