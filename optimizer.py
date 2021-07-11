import numpy as np

class ScheduledOptim():
    ''' A simple wrapper class for learning rate scheduling '''

    def __init__(self, optimizer, d_model, n_warmup_steps, current_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = current_steps
        self.init_lr = np.power(d_model, -0.5)

    def state_dict(self):
        return self._optimizer.state_dict()

    def step(self):
        self._optimizer.step()

    def step_and_update_lr(self, step=None):
        self._update_learning_rate(step=step)
        self._optimizer.step()

    def zero_grad(self):
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self, step=None):
        ''' Learning rate scheduling per step '''
        if step is None:
            self.n_current_steps += 1
        else:
            self.n_current_steps += step
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
