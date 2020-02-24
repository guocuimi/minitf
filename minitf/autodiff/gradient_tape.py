from minitf import kernel as K
from minitf.autodiff.graph import Graph
from minitf.autodiff.graph import pop_graph
from minitf.autodiff.graph import push_graph
from minitf.autodiff.graph import toposort


class GradientTape(object):
    def __init__(self):
        self._graph = Graph()

    def __enter__(self):
        push_graph(self._graph)
        return self

    def __exit__(self, *args):
        pop_graph()

    @staticmethod
    def _accumulate_grad(prev_g, curr_g):
        if prev_g is None:
            return curr_g
        return prev_g + curr_g

    def gradient(self, target, sources):
        outgrads = {target: K.ones_like(target)}

        for node, neighbors, jvps in toposort(self._graph, target):
            parent_grad = outgrads[node]
            for neighbor, jvp in zip(neighbors, jvps):
                outgrads[neighbor] = self._accumulate_grad(
                    outgrads.get(neighbor), jvp(parent_grad))
        if isinstance(sources, (list, tuple)):
            return [outgrads.get(s) for s in sources]
        return outgrads.get(sources)
