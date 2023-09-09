from dataclasses import dataclass, field
from typing import List, Union

import numpy
import numpy as np
import torch

from .utils.chebyshev_ball import chebyshev_ball


@dataclass
class CriticalRegion:
    r"""
    Critical region is a polytope that defines a region in the uncertainty space
    with an associated optimal value, active set, lagrange multipliers and
    constraints

    .. math::

        \begin{align}
            x(\theta) &= A\theta + b\\
            \lambda(\theta) &= C\theta + d\\
            \Theta &:= \{\forall \theta \in \mathbf{R}^m: E\theta \leq f\}
        \end{align}

    equality_indices: numpy array of indices

    constraint_set: if this is an A@x = b + F@theta boundary

    lambda_set: if this is a λ = 0 boundary

    boundary_set: if this is an Eθ <= f boundary

    """

    A: numpy.ndarray
    b: numpy.ndarray
    C: numpy.ndarray
    d: numpy.ndarray
    E: numpy.ndarray
    f: numpy.ndarray
    active_set: Union[List[int], numpy.ndarray]

    omega_set: Union[List[int], numpy.ndarray] = field(default_factory=list)
    lambda_set: Union[List[int], numpy.ndarray] = field(default_factory=list)
    regular_set: Union[List[int], numpy.ndarray] = field(default_factory=list)

    y_fixation: numpy.ndarray = None
    y_indices: numpy.ndarray = None
    x_indices: numpy.ndarray = None

    def __repr__(self):
        """Returns a String output of Critical Region."""
        return f"Critical region with active set {self.active_set}\nThe Omega Constraint indices are {self.omega_set}\nThe Lagrange multipliers Constraint indices are {self.lambda_set}\nThe Regular Constraint indices are {self.regular_set}\n  x(θ) = Aθ + b \n λ(θ) = Cθ + d \n  Eθ <= f \n A = {self.A} \n b = {self.b} \n C = {self.C} \n d = {self.d} \n E = {self.E} \n f = {self.f}"

    def evaluate(self, theta: numpy.ndarray) -> numpy.ndarray:
        """Evaluates x(θ) = Aθ + b."""

        if self.y_fixation is not None:
            cont_vars = self.A @ theta + numpy.expand_dims(self.b, axis=1)
            x_star = numpy.zeros((len(self.x_indices) + len(self.y_indices),))
            x_star[self.x_indices] = cont_vars.flatten()
            x_star[self.y_indices] = self.y_fixation
            return x_star.reshape(-1, 1)
        else:
            if len(self.b.shape) == 1:
                return self.A @ theta + self.b.reshape(-1, 1)
            else:
                return self.A @ theta + self.b

    def lagrange_multipliers(self, theta: numpy.ndarray) -> numpy.ndarray:
        """Evaluates λ(θ) = Cθ + d."""
        return self.C @ theta + self.d

    def is_inside(self, theta: numpy.ndarray) -> numpy.ndarray:
        """Tests if point θ is inside the critical region."""
        # print(f"self.E @ theta - self.f = {self.E @ theta - self.f}")
        return numpy.all(self.E @ theta - self.f <= 0.01)   #  改成 <= 0 了 2023.8.2；改回 < 0 23.8.8\

    def distance(self, theta: numpy.ndarray) -> dict:
        """Calculate the smallest distance between point θ and the critical regions."""
        smallest_distance = 0
        # 计算点到超平面的距离, 使用矩阵的方式计算。代码如下：  23.09.02
        # Calculate distances using matrix operations
        distances = self.E @ theta - self.f\
                    # / np.linalg.norm(self.E, axis=1)

        # Find the maximum distance
        smallest_distance = np.max(distances)
        idx = np.argmax(distances)
        if self.lambda_set:
            if isinstance(self.lambda_set, torch.Tensor):
                lambda_list = [item.cpu().detach().item() for item in self.lambda_set]
            elif self.lambda_set is []:
                lambda_list = []
            elif isinstance(self.lambda_set[0], numpy.ndarray):
                lambda_list = [item.item() for item in self.lambda_set]
            else:
                lambda_list = self.lambda_set
        else:
            lambda_list = []
        sm_dis_dict = {'smallest_distance': smallest_distance, 'idx': idx,'length of lambda set': len(self.lambda_set), 'lambda set idx': lambda_list}

        return sm_dis_dict

    # depreciated
    def is_full_dimension(self) -> bool:
        """Tests dimensionality of critical region."""
        # I think so

        soln = chebyshev_ball(self.E, self.f)
        if soln is not None:
            return soln.sol[-1] > 10 ** -8
        return False

    def get_constraints(self):
        return [self.E, self.f]

    # def __getstate__(self):
    #     # Define what attributes need to be serialized
    #     state = self.__dict__.copy()
    #     # Remove any attribute that should not be pickled
    #     # For example, you might remove complex objects or references to other objects
    #     # del state['attribute_name']
    #     return state
    #
    # def __setstate__(self, state):
    #     # Restore the attributes during deserialization
    #     self.__dict__.update(state)
    #     # Additional initialization code if needed
