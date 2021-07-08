import torch
from torch.autograd import Function
from torch.nn.functional import linear
from torch import Tensor


class RotationalLinearFunction(Function):
    @staticmethod
    def forward(ctx, *args) -> Tensor:
        x, w, b, learn_left, learn_right = args
        learn_l = torch.as_tensor(learn_left).requires_grad_(False)
        learn_r = torch.as_tensor(learn_right).requires_grad_(False)

        ctx.save_for_backward(x, w, b, learn_l, learn_r)
        return linear(x, w, b)

    @staticmethod
    def backward(ctx, *args: Tensor) -> tuple:
        raise NotImplementedError


class RotationalLinearFunctionByInsertingGrad(RotationalLinearFunction):
    @staticmethod
    def backward(ctx, *args: Tensor) -> tuple:
        grad = args[0]
        x, w, b, learn_l, learn_r = ctx.saved_tensors
        w = w.t()
        learn_l, learn_r = int(learn_l), int(learn_r)

        # バイアスへの勾配は、0ベクトルを作って必要な要素だけ値を入れる
        # gradients for bias, make 0 vector and insert value into needed element
        if grad.is_cuda:
            d_b = torch.zeros(size=(grad.shape[1],), dtype=torch.float32, device='cuda')
        else:
            d_b = torch.zeros(size=(grad.shape[1],), dtype=torch.float32)
        d_b[learn_l:learn_r] = torch.sum(grad[:, learn_l:learn_r], dim=0)

        # 重みへの勾配は、0行列を作って必要な行だけ値を入れる
        # gradients for weights, make 0 matrix and insert value into needed column
        if grad.is_cuda:
            d_w = torch.zeros(size=(x.shape[1], grad.shape[1]), dtype=torch.float32, device='cuda')
        else:
            d_w = torch.zeros(size=(x.shape[1], grad.shape[1]), dtype=torch.float32)
        d_w[:, learn_l:learn_r] = torch.matmul(x.t(), grad[:, learn_l:learn_r])

        d_x = torch.matmul(grad, torch.t(w))
        return d_x, d_w.t(), d_b, None, None


class RotationalLinearFunctionByInsertingZero(RotationalLinearFunction):
    @staticmethod
    def forward(ctx, *args) -> Tensor:
        x, w, b, learn_left, learn_right = args
        learn_l = torch.as_tensor(learn_left).requires_grad_(False)
        learn_r = torch.as_tensor(learn_right).requires_grad_(False)

        ctx.save_for_backward(x, w, b, learn_l, learn_r)
        return linear(x, w, b)

    @staticmethod
    def backward(ctx, *args: Tensor) -> tuple:
        grad = args[0]
        x, w, b, learn_l, learn_r = ctx.saved_tensors
        w = w.t()
        learn_l, learn_r = int(learn_l), int(learn_r)

        # バイアスへの勾配は、学習しない部分に0を入れる
        # gradients for bias, insert 0 into removed element
        d_b = torch.sum(grad, dim=0)
        d_b[:learn_l] = d_b[learn_r:] = 0

        # 重みへの勾配は、学習しない列に0を入れる
        # gradients for weights, insert 0 into removed column
        d_w = torch.matmul(x.t(), grad)
        d_w[:, :learn_l] = d_w[:, learn_r:] = 0

        d_x = torch.matmul(grad, torch.t(w))
        return d_x, d_w.t(), d_b, None, None
