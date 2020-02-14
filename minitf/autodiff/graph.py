from .util import to_list


class Graph(object):
    def __init__(self):
        self._edges = dict()

    def add_edges(self, source, targets, metadata=None):
        self._edges[source] = (to_list(targets), metadata)

    def get_targets(self, source, default=(None, None)):
        return self._edges.get(source, default)


_GRAPH_STACK = []


def get_current_graph():
    if _GRAPH_STACK:
        return _GRAPH_STACK[-1]
    return None


def push_graph(graph):
    _GRAPH_STACK.append(graph)


def pop_graph():
    return _GRAPH_STACK.pop()


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
        neighbors, jvps = graph.get_targets(node, ([], []))
        yield node, neighbors, jvps
        for neighbor in neighbors:
            if child_counts[neighbor] == 1:
                childless_nodes.append(neighbor)
            else:
                child_counts[neighbor] -= 1
