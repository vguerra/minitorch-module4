from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


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
    val = f(*vals)
    vals_epsilon = list(vals)
    vals_epsilon[arg] += epsilon
    val_delta = f(*vals_epsilon)
    f_prime = (val_delta - val) / epsilon
    return f_prime


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
    queue = [variable]
    sorted_vars = []
    visited = set()

    while queue:
        var = queue[0]
        queue.pop(0)
        if var.unique_id not in visited:
            if var.is_constant():
                continue
            visited.add(var.unique_id)
            sorted_vars.append(var)
            for parent in var.parents:
                queue.append(parent)

    return sorted_vars


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    vars = list(topological_sort(variable))
    tracked_deriv: Dict[int, Any] = {variable.unique_id: deriv}

    while vars:
        var = vars[0]
        vars.pop(0)
        current_deriv = tracked_deriv[var.unique_id]
        if var.is_leaf():
            var.accumulate_derivative(current_deriv)
        else:
            for (input_var, deriv_out) in var.chain_rule(current_deriv):
                tracked_deriv[input_var.unique_id] = (
                    tracked_deriv.get(input_var.unique_id, 0.0) + deriv_out
                )


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
