import torch
import torch.nn.functional as F
import torch.nn as nn


class MLP(torch.nn.Module):
    def __init__(self, in_size, out_size, hidde_size=(64, ), activ_funct='relu', outpu='probs', init_type='uniform'):
        super(MLP, self).__init__()
        self.__in_size = in_size
        self.__layes = torch.nn.ModuleDict()
        self.__hidde_num = len(hidde_size)
        self.__activ_funct = activ_funct
        self.__outpu = outpu

        # input layer
        _first_layer = torch.nn.Linear(in_size, hidde_size[0])

        if init_type == 'uniform':
            nn.init.kaiming_uniform_(_first_layer.weight)
        elif init_type == 'normal':
            nn.init.kaiming_normal_(_first_layer.weight)
        elif init_type == 'default':
            pass

        self.__layes['layer_0'] = _first_layer
        # hidden layers
        for l in range(self.__hidde_num-1):
            _hidde_lay = torch.nn.Linear(hidde_size[l], hidde_size[l+1])

            if init_type == 'uniform':
                nn.init.kaiming_uniform_(_hidde_lay.weight)
            elif init_type == 'normal':
                nn.init.kaiming_normal_(_hidde_lay.weight)
            elif init_type == 'default':
                pass

            self.__layes['layer_{}'.format(l+1)] = _hidde_lay
        # output layer
        _last_layer = torch.nn.Linear(hidde_size[-1], out_size)

        if init_type == 'uniform':
            nn.init.kaiming_uniform_(_last_layer.weight)
        elif init_type == 'normal':
            nn.init.kaiming_normal_(_last_layer.weight)
        elif init_type == 'default':
            pass

        self.__layes['layer_{}'.format(self.__hidde_num)] = _last_layer

    def forward(self, x):
        if type(x) == list:
            x = torch.tensor(
                x, dtype=torch.float32).reshape(-1, self.__in_size)
        for l in range(self.__hidde_num):
            layer = self.__layes['layer_{}'.format(l)]
            x = layer(x)
            if self.__activ_funct == 'relu':
                x = F.relu(x)
        logit = self.__layes['layer_{}'.format(self.__hidde_num)](x)
        if self.__outpu == 'probs':
            probs = F.softmax(logit, dim=1)
            return probs
        elif self.__outpu == 'logits':
            return logit
        elif self.__outpu == 'sigmoid':
            return torch.sigmoid(logit)


if __name__ == '__main__':
    mlp = MLP(10, 2, hidde_size=(64, 32), activ_funct='relu',
              outpu='probs', init_type='uniform')
    x = torch.randn(1, 10)
    a = mlp(x)
