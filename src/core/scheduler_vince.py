#2025.02.26
# 根据实验数据观察到 本次模型对学习率异常敏感，所以需要设计一个 动态的 学习率衰减方式
# 学习率可以 根据 epoch衰减，也可以根据 batch 衰减
# 接下来我将通过 这两种衰减方式 实现一个兼容的LR Scheduler
import torch
from torch.optim import Optimizer
from torch import Tensor
import contextlib
import logging
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

class LRScheduler(object):
    def __init__(self, optimizer: Optimizer, verbose = False):
        if not isinstance(optimizer, Optimizer): # 先判断是不是 optimizer
            raise TypeError('{} is not an Optimizer.'.format(type(optimizer).__name__))
        self.optimizer = optimizer
        self.verbose = verbose

        for group in optimizer.param_groups:
            group.setdefault('base_lr', group['lr'])  #为所有的参数组设置一个key：base_lr, value初始值是 group['lr']， 记录一下base的lr，方便后面进行操作

        self.base_lrs = [group['base_lr'] for group in optimizer.param_groups]

        self.epoch = 0
        self.batch = 0

    def state_dict(self):
        return {
            'epoch':self.epoch,
            'batch':self.batch,
        }
    def load_state_dict(self, state_dict):
        base_lrs = self.base_lrs
        self.__dict__.update(state_dict)  #使用update方法将state_dict中的键值对更新到当前对象的属性
        self.base_lrs = base_lrs

    def get_last_lr(self):
        return self._last_lr

    def get_lr(self):
        raise NotImplementedError #这里先没必要实现

    def step_batch(self, batch = None):  #根据 batch 来 迭代
        if batch is not None:
            self.batch = batch
        else:
            self.batch = self.batch + 1
        self._set_lrs()

    def step_epoch(self, epoch = None):
        if epoch is not None:
            self.epoch = epoch
        else:
            self.epoch = self.epoch + 1
        self._set_lrs()

    def _set_lrs(self):
        values = self.get_lr()
        assert len(values) == len(self.optimizer.param_groups)

        for index, data in enumerate(zip(self.optimizer.param_groups, values)): #一一对应的
            param_group, lr = data
            param_group['lr'] = lr  #更新学习率
            self.print_lr(self.verbose, index, lr)
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def print_lr(self, is_verbose, group, lr):
        if is_verbose:
            print(f'epoch={self.epoch}, batch = {self.batch} : adjusting learning rate of group {group} to {lr:.4e}' )

class Vince_scheduler(LRScheduler):
    def __init__(self,
                 optimizer: Optimizer,
                 lr_batches: Union[int,float],
                 lr_epochs: Union[int,float],
                 warmup_epochs: Union[int, float] = 10,
                 warmup_start: float = 0.7,
                 verbose: bool = False):
        super(Vince_scheduler, self).__init__(optimizer, verbose)
        self.lr_batches = lr_batches
        self.lr_epochs = lr_epochs
        self.warmup_epochs = warmup_epochs

        assert 0.0 <= warmup_start <= 1.0, warmup_start
        self.warmup_start = warmup_start
        print(f'using Vince_scheduler:lr_epoches:{self.lr_epochs}, warmup_epochs:{self.warmup_epochs}, warmup_start:{self.warmup_start}')
    def get_lr(self):


        warmup_factor = (
            1.0
            if self.epoch >= self.warmup_epochs
            else self.warmup_start
            + (1.0 - self.warmup_start) * (self.epoch / self.warmup_epochs)
            # else 0.5 + 0.5 * (self.batch / self.warmup_batches)
        )
        if warmup_factor == 1.0:
            factor = (
                ((self.epoch**2 + self.lr_epochs**2) / self.lr_epochs**2) ** -0.25
            )
        else:
            factor = 1
        print(f'factor:{factor}, warmup_factor:{warmup_factor}, lr_epochs:{self.lr_epochs}, epoch: {self.epoch}')
        print(f'current lr: {[x * factor * warmup_factor for x in self.base_lrs]}')
        return [x * factor * warmup_factor for x in self.base_lrs]

#一个新的优化器 eve：

class Eve(Optimizer):
    """
    Implements Eve algorithm.  This is a modified version of AdamW with a special
    way of setting the weight-decay / shrinkage-factor, which is designed to make the
    rms of the parameters approach a particular target_rms (default: 0.1).  This is
    for use with networks with 'scaled' versions of modules (see scaling.py), which
    will be close to invariant to the absolute scale on the parameter matrix.

    The original Adam algorithm was proposed in `Adam: A Method for Stochastic Optimization`_.
    The AdamW variant was proposed in `Decoupled Weight Decay Regularization`_.
    Eve is unpublished so far.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 3e-4;
            this value means that the weight would decay significantly after
            about 3k minibatches.  Is not multiplied by learning rate, but
            is conditional on RMS-value of parameter being > target_rms.
        target_rms (float, optional): target root-mean-square value of
           parameters, if they fall below this we will stop applying weight decay.


    .. _Adam: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=1e-3,
        target_rms=0.1,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0 <= weight_decay <= 0.1:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0 < target_rms <= 10.0:
            raise ValueError("Invalid target_rms value: {}".format(target_rms))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            target_rms=target_rms,
        )
        super(Eve, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Eve, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Perform optimization step
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                beta1, beta2 = group["betas"]

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = (exp_avg_sq.sqrt() * (bias_correction2**-0.5)).add_(
                    group["eps"]
                )

                step_size = group["lr"] / bias_correction1
                target_rms = group["target_rms"]
                weight_decay = group["weight_decay"]

                if p.numel() > 1:
                    # avoid applying this weight-decay on "scaling factors"
                    # (which are scalar).
                    is_above_target_rms = p.norm() > (target_rms * (p.numel() ** 0.5))
                    p.mul_(1 - (weight_decay * is_above_target_rms))

                p.addcdiv_(exp_avg, denom, value=-step_size)

                if random.random() < 0.0005:
                    step = (exp_avg / denom) * step_size
                    logging.info(
                        f"Delta rms = {(step**2).mean().item()}, shape = {step.shape}"
                    )

        return loss










