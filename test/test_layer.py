from copy import deepcopy
import unittest

import torch
from torch import tensor
from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn.modules import Linear
from torchvision import models

from rotational_update import RotationalLinear


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


class TestRotationalLinearNet(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.vgg16 = models.vgg16().to(device)
        self.vgg16.classifier[-1] = Linear(4096, 10, bias=True).to(device)
        self.rotated_vgg16 = deepcopy(self.vgg16)

    # 同期式のレイヤーから、パラメータをそのままにして準同期式レイヤを作る
    def testParam(self):
        self.rotated_vgg16.classifier[0] = RotationalLinear(
            self.rotated_vgg16.classifier[0],
        ).to(device)
        self.rotated_vgg16.classifier[3] = RotationalLinear(
            self.rotated_vgg16.classifier[3],
        ).to(device)

        # 別オブジェクトだが同じパラメータを持つことを確認
        self.assertTrue(self.vgg16 is not self.rotated_vgg16)

        for layer1, layer2 in zip(self.vgg16.classifier, self.rotated_vgg16.classifier):
            self.assertTrue(isinstance(layer1, Linear) == isinstance(layer2, Linear))

            if isinstance(layer1, Linear) and isinstance(layer2, Linear):
                self.assertTrue((layer1.weight == layer2.weight).byte().all())
                self.assertTrue((layer1.bias == layer2.bias).byte().all())


class TestRotationalLinear(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        base_layer = Linear(2, 4)
        base_layer.weight = Parameter(tensor([
            [1.7, 0.4, 1, 2.2],
            [1.8, -1, 0.9, -0.2]
        ], requires_grad=True).t())
        base_layer.bias = Parameter(tensor([0.0, 0.0, 0.0, 0.0], requires_grad=True))

        self.layer = RotationalLinear(base_layer).to(device)

    # def testGroup(self):
    #     self.assertEqual([0, 2, 4], self.layer.group_delim)

    def testForward(self):
        X = tensor([  # データ数3と仮定, shape=(3, 2)
            [1.2, 1.4],
            [-0.8, 0.2],
            [1.1, -1.2]
        ]).to(device)
        expected = tensor([
            [4.56, -0.92, 2.46, 2.36],
            [-1, -0.52, -0.62, -1.8],
            [-0.29, 1.64, 0.02, 2.66]
        ]).to(device)
        actual = self.layer.forward(X)
        self.assertTrue(torch.all(torch.abs(expected - actual) < 10e-6))

    def testBackward(self) -> None:
        X = Variable(tensor([  # データ数3と仮定, shape=(3, 2)
            [1.2, 1.4],
            [-0.8, 0.2],
            [1.1, -1.2]
        ], requires_grad=True).to(device))
        X.requires_grad_()

        expected_dx = tensor([
            [5.3, 1.5],
            [5.3, 1.5],
            [5.3, 1.5]
        ]).to(device)

        # -- neuron 0,1 get gradient --
        expected_dW = tensor([
            [1.5, 1.5, 0, 0],
            [0.4, 0.4, 0, 0]
        ]).t().to(device)
        expected_db = tensor([3.0, 3.0, 0, 0]).to(device)

        dout = self.layer.forward(X)
        dout.backward(torch.ones((3, 4)).to(device))

        self.assertTrue(
            torch.all(torch.abs(expected_dx - X.grad) < 10e-6).item())
        self.assertTrue(
            torch.all(torch.abs(expected_dW - self.layer.weight.grad) < 10e-6).item())
        self.assertTrue(
            torch.all(torch.abs(expected_db - self.layer.bias.grad) < 10e-6).item())

        self.layer.rotate()

        # clear gradients
        self.layer.weight.grad = None
        self.layer.bias.grad = None
        X.grad = None

        # -- neuron 2,3 get gradient --
        expected_dW = tensor([
            [0, 0, 1.5, 1.5],
            [0, 0, 0.4, 0.4]
        ]).t().to(device)
        expected_db = tensor([0, 0, 3.0, 3.0]).to(device)

        dout = self.layer.forward(X)
        dout.backward(torch.ones((3, 4)).to(device))

        self.assertTrue(
            torch.all(torch.abs(expected_dx - X.grad) < 10e-6).item())
        self.assertTrue(
            torch.all(torch.abs(expected_dW - self.layer.weight.grad) < 10e-6).item())
        self.assertTrue(
            torch.all(torch.abs(expected_db - self.layer.bias.grad) < 10e-6).item())

        self.layer.rotate()

        # clear gradients
        self.layer.weight.grad = None
        self.layer.bias.grad = None
        X.grad = None

        # -- neuron 0,1 get gradient again --
        expected_dW = tensor([
            [1.5, 1.5, 0, 0],
            [0.4, 0.4, 0, 0]
        ]).t().to(device)
        expected_db = tensor([3.0, 3.0, 0, 0]).to(device)

        dout = self.layer.forward(X)
        dout.backward(torch.ones((3, 4)).to(device))
        self.assertTrue(
            torch.all(torch.abs(expected_dx - X.grad) < 10e-6).item())
        self.assertTrue(
            torch.all(torch.abs(expected_dW - self.layer.weight.grad) < 10e-6).item())
        self.assertTrue(
            torch.all(torch.abs(expected_db - self.layer.bias.grad) < 10e-6).item())

    def testRotate(self):
        X = tensor([  # データ数3と仮定, shape=(3, 2)
                    [1.2, 1.4],
                    [-0.8, 0.2],
                    [1.1, -1.2]
        ], requires_grad=True).to(device)

        self.layer.forward(X)
        left = self.layer.learn_l
        right = self.layer.learn_r
        self.assertEqual(0, left)
        self.assertEqual(2, right)
        self.layer.rotate()

        self.layer.forward(X)
        left = self.layer.learn_l
        right = self.layer.learn_r
        self.assertEqual(2, left)
        self.assertEqual(4, right)
        self.layer.rotate()

        self.layer.forward(X)
        left = self.layer.learn_l
        right = self.layer.learn_r
        self.assertEqual(0, left)
        self.assertEqual(2, right)
        self.layer.rotate()


class TestRotationalLinearAtUnbalance(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.layer = RotationalLinear(Linear(4000, 4000)).to(device)

    def testGroup(self):
        expected = [63 * i for i in range(64)] + [4000]
        self.assertEqual(expected, self.layer.group_partition)

    def testRotate(self):
        X = torch.ones((1, 4000)).to(device)

        for i in range(63):
            self.layer.forward(X)
            left = self.layer.learn_l
            right = self.layer.learn_r
            self.assertEqual(63 * i, left)
            self.assertEqual(63 * (i + 1), right)
            self.layer.rotate()

        self.layer.forward(X)
        left = self.layer.learn_l
        right = self.layer.learn_r
        self.assertEqual(3969, left)
        self.assertEqual(4000, right)
        self.layer.rotate()

        self.layer.forward(X)
        left = self.layer.learn_l
        right = self.layer.learn_r
        self.assertEqual(0, left)
        self.assertEqual(63, right)
        self.layer.rotate()


# Test for Non Reduce Backward Version
class TestRotationalLinearNetNonReduceBackward(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.vgg16 = models.vgg16().to(device)
        self.vgg16.classifier[-1] = Linear(4096, 10, bias=True).to(device)
        self.rotated_vgg16 = deepcopy(self.vgg16)

    # 同期式のレイヤーから、パラメータをそのままにして準同期式レイヤを作る
    def testParam(self):
        self.rotated_vgg16.classifier[0] = RotationalLinear(
            self.rotated_vgg16.classifier[0],
            reduce_backward=False
        ).to(device)
        self.rotated_vgg16.classifier[3] = RotationalLinear(
            self.rotated_vgg16.classifier[3],
            reduce_backward=False
        ).to(device)

        # 別オブジェクトだが同じパラメータを持つことを確認
        self.assertTrue(self.vgg16 is not self.rotated_vgg16)

        for layer1, layer2 in zip(self.vgg16.classifier, self.rotated_vgg16.classifier):
            self.assertTrue(isinstance(layer1, Linear) == isinstance(layer2, Linear))

            if isinstance(layer1, Linear) and isinstance(layer2, Linear):
                self.assertTrue((layer1.weight == layer2.weight).byte().all())
                self.assertTrue((layer1.bias == layer2.bias).byte().all())


class TestRotationalLinearNonReduceBackward(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        base_layer = Linear(2, 4)
        base_layer.weight = Parameter(tensor([
            [1.7, 0.4, 1, 2.2],
            [1.8, -1, 0.9, -0.2]
        ], requires_grad=True).t())
        base_layer.bias = Parameter(tensor([0.0, 0.0, 0.0, 0.0], requires_grad=True))

        self.layer = RotationalLinear(base_layer, reduce_backward=False).to(device)

    # def testGroup(self):
    #     self.assertEqual([0, 2, 4], self.layer.group_delim)

    def testForward(self):
        X = tensor([  # データ数3と仮定, shape=(3, 2)
            [1.2, 1.4],
            [-0.8, 0.2],
            [1.1, -1.2]
        ]).to(device)
        expected = tensor([
            [4.56, -0.92, 2.46, 2.36],
            [-1, -0.52, -0.62, -1.8],
            [-0.29, 1.64, 0.02, 2.66]
        ]).to(device)
        actual = self.layer.forward(X)
        self.assertTrue(torch.all(torch.abs(expected - actual) < 10e-6))

    def testBackward(self) -> None:
        X = Variable(tensor([  # データ数3と仮定, shape=(3, 2)
            [1.2, 1.4],
            [-0.8, 0.2],
            [1.1, -1.2]
        ], requires_grad=True).to(device))
        X.requires_grad_()

        expected_dx = tensor([
            [5.3, 1.5],
            [5.3, 1.5],
            [5.3, 1.5]
        ]).to(device)

        # -- neuron 0,1 get gradient --
        expected_dW = tensor([
            [1.5, 1.5, 0, 0],
            [0.4, 0.4, 0, 0]
        ]).t().to(device)
        expected_db = tensor([3.0, 3.0, 0, 0]).to(device)

        dout = self.layer.forward(X)
        dout.backward(torch.ones((3, 4)).to(device))

        self.assertTrue(
            torch.all(torch.abs(expected_dx - X.grad) < 10e-6).item())
        self.assertTrue(
            torch.all(torch.abs(expected_dW - self.layer.weight.grad) < 10e-6).item())
        self.assertTrue(
            torch.all(torch.abs(expected_db - self.layer.bias.grad) < 10e-6).item())

        self.layer.rotate()

        # clear gradients
        self.layer.weight.grad = None
        self.layer.bias.grad = None
        X.grad = None

        # -- neuron 2,3 get gradient --
        expected_dW = tensor([
            [0, 0, 1.5, 1.5],
            [0, 0, 0.4, 0.4]
        ]).t().to(device)
        expected_db = tensor([0, 0, 3.0, 3.0]).to(device)

        dout = self.layer.forward(X)
        dout.backward(torch.ones((3, 4)).to(device))

        self.assertTrue(
            torch.all(torch.abs(expected_dx - X.grad) < 10e-6).item())
        self.assertTrue(
            torch.all(torch.abs(expected_dW - self.layer.weight.grad) < 10e-6).item())
        self.assertTrue(
            torch.all(torch.abs(expected_db - self.layer.bias.grad) < 10e-6).item())

        self.layer.rotate()

        # clear gradients
        self.layer.weight.grad = None
        self.layer.bias.grad = None
        X.grad = None

        # -- neuron 0,1 get gradient again --
        expected_dW = tensor([
            [1.5, 1.5, 0, 0],
            [0.4, 0.4, 0, 0]
        ]).t().to(device)
        expected_db = tensor([3.0, 3.0, 0, 0]).to(device)

        dout = self.layer.forward(X)
        dout.backward(torch.ones((3, 4)).to(device))
        self.assertTrue(
            torch.all(torch.abs(expected_dx - X.grad) < 10e-6).item())
        self.assertTrue(
            torch.all(torch.abs(expected_dW - self.layer.weight.grad) < 10e-6).item())
        self.assertTrue(
            torch.all(torch.abs(expected_db - self.layer.bias.grad) < 10e-6).item())

    def testRotate(self):
        X = tensor([  # データ数3と仮定, shape=(3, 2)
                    [1.2, 1.4],
                    [-0.8, 0.2],
                    [1.1, -1.2]
        ], requires_grad=True).to(device)

        self.layer.forward(X)
        left = self.layer.learn_l
        right = self.layer.learn_r
        self.assertEqual(0, left)
        self.assertEqual(2, right)
        self.layer.rotate()

        self.layer.forward(X)
        left = self.layer.learn_l
        right = self.layer.learn_r
        self.assertEqual(2, left)
        self.assertEqual(4, right)
        self.layer.rotate()

        self.layer.forward(X)
        left = self.layer.learn_l
        right = self.layer.learn_r
        self.assertEqual(0, left)
        self.assertEqual(2, right)
        self.layer.rotate()


class TestRotationalLinearAtUnbalanceNonReduceBackward(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.layer = RotationalLinear(Linear(4000, 4000), reduce_backward=False).to(device)

    def testGroup(self):
        expected = [63 * i for i in range(64)] + [4000]
        self.assertEqual(expected, self.layer.group_partition)

    def testRotate(self):
        X = torch.ones((1, 4000)).to(device)

        for i in range(63):
            self.layer.forward(X)
            left = self.layer.learn_l
            right = self.layer.learn_r
            self.assertEqual(63 * i, left)
            self.assertEqual(63 * (i + 1), right)
            self.layer.rotate()

        self.layer.forward(X)
        left = self.layer.learn_l
        right = self.layer.learn_r
        self.assertEqual(3969, left)
        self.assertEqual(4000, right)
        self.layer.rotate()

        self.layer.forward(X)
        left = self.layer.learn_l
        right = self.layer.learn_r
        self.assertEqual(0, left)
        self.assertEqual(63, right)
        self.layer.rotate()


if __name__ == '__main__':
    unittest.main()
