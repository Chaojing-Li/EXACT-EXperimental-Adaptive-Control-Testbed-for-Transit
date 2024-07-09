from typing import Tuple

import torch
from .approximator import MLP


class Actor_Net(torch.nn.Module):
    def __init__(self, state_size: int, hidde_size: Tuple, init_type: str = 'default'):
        ''' Actor network with 

        Args:
            state_size: size of the state (backward and forward spacing, and discrete state if any)
            hidde_size: tuple with the number of neurons in each hidden layer for continuous feature
        '''
        super(Actor_Net, self).__init__()
        self._mlp = MLP(state_size, 1, hidde_size,
                        outpu='sigmoid', init_type=init_type)

    def forward(self, x):
        # if type(x) == tuple:
        # x = torch.tensor(x, dtype=torch.float32).reshape(-1, 1)
        return self._mlp(x)


class Critic_Net(torch.nn.Module):
    def __init__(self, state_size, hidde_size, init_type='default'):
        self._state_size = state_size
        super(Critic_Net, self).__init__()
        self._mlp = MLP(self._state_size+1, 1, hidde_size,
                        outpu='logits', init_type=init_type)

    def forward(self, x):
        s = x[:, 0:self._state_size]
        a = x[:, self._state_size].unsqueeze(1)
        conti_x_a = torch.cat((s, a), dim=1)
        logit = self._mlp(conti_x_a)
        return logit
