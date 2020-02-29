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

    def gradient(self, target, sources, output_grad=None):
        if output_grad is None:
            output_grad = K.ones_like(target)
        out_grads = {target: output_grad}

        for node, neighbors, vjps in toposort(self._graph, target):
            parent_grad = out_grads[node]
            for neighbor, vjp in zip(neighbors, vjps):
                out_grads[neighbor] = self._accumulate_grad(
                    out_grads.get(neighbor), vjp(parent_grad))
        if isinstance(sources, (list, tuple)):
            return type(sources)(out_grads.get(s) for s in sources)
        return out_grads.get(sources)
