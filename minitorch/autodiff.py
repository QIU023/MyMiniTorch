from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


from copy import deepcopy

def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # TODO: Implement for Task 1.1.
    # raise NotImplementedError('Need to implement for Task 1.1')
    # grads = []
    # for i in range(arg):
    #     new_val = deepcopy(vals)
    #     new_val[i] += epsilon
    #     obj_diff = f(new_val) - f(vals)
    #     grads.append(obj_diff / epsilon)
    # return grads
    tmp_val = list(vals[:])
    tmp_val[arg] += epsilon/2
    forward_value = f(*tmp_val)
    tmp_val[arg] -= epsilon
    backward_value = f(*tmp_val)

    return (forward_value-backward_value) / epsilon


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.
    # raise NotImplementedError('Need to implement for Task 1.4')
    return_list = []
    seen_set = set()
    # view_var = {}

    def dfs(v: Variable):
        if v.is_constant():
            return
        seen_set.add(v.unique_id)
        # view_var[v.unique_id] = 1
        return_list.append(v)
        for vi in v.parents:
            if seen_set.contains(vi.unique_id):
            # if view_var.get(vi.unique_id, 0):
                continue
            dfs(vi)
    dfs(variable)
    return return_list


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    # raise NotImplementedError('Need to implement for Task 1.4')
    queue = [(variable, deriv)]
    while len(queue) > 0:
        v, d = queue.pop(0)
        if v.is_leaf():
            v.accumulate_derivative(d)
        else:
            parent_vars = v.chain_rule(d)
            queue.extend(parent_vars)


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
