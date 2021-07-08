from copy import deepcopy
from math import sqrt

from torch import Tensor
from torch.nn.modules import Linear

from rotational_update.layers.base import Rotatable
from rotational_update.layers.functions.static import RotationalLinearFunctionByInsertingGrad
from rotational_update.layers.functions.static import RotationalLinearFunctionByInsertingZero


class RotationalLinear(Linear, Rotatable):
    """
    Rotatable Linear class
    """
    def __init__(self, linear: Linear, reduce_backward: bool = True):
        """
        Parameters
        ----------
        linear : torch.nn.modules.Linear
            Base Linear object. RotatationalLinear object takes over weights and biases from base object.
        """
        super().__init__(linear.in_features, linear.out_features, bias=True)
        self.reduce_backward = reduce_backward

        # 重みを引き継ぎ
        # use current weights
        self.weight = deepcopy(linear.weight)
        self.bias = deepcopy(linear.bias)
        self.learn_l = None
        self.learn_r = None

        output_features = linear.out_features

        # group size -> sqrt(output_features)
        num = int(sqrt(output_features))
        group_list = [num for _ in range(num)]
        nueron_sum = sum(group_list)

        if nueron_sum < output_features:  # make one more group with extra neurons
            group_list.append(output_features - nueron_sum)

        # cumulative sum for slicing
        group_partition = [0] + group_list
        for i in range(1, len(group_partition)):
            group_partition[i] += group_partition[i - 1]

        self.group_partition = group_partition
        self.group_i = 1

    def forward(self, input_tensor) -> Tensor:
        """
        Feed-forward method
        Almost same to normal Linear object.
        Save variables for learning group.

        Parameters
        ----------
        input_tensor : torch.Tensor

        Returns
        -------
        res : Output of feed-forwarding.
        """
        self.learn_l = self.group_partition[self.group_i-1]
        self.learn_r = self.group_partition[self.group_i]

        if self.reduce_backward:
            matmul = RotationalLinearFunctionByInsertingGrad.apply
        else:
            matmul = RotationalLinearFunctionByInsertingZero.apply
        res = matmul(input_tensor, self.weight, self.bias, self.learn_l, self.learn_r)

        return res

    def rotate(self):
        """
        Change learning group to next one.
        Call this at every minibatch spending.
        """
        self.group_i += 1
        if self.group_i == len(self.group_partition):
            self.group_i = 1  # 最初のグループへ  to initial group
