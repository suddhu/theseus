import torch
import torch.nn as nn


def vec2ss_matrix(vector):  # vector to skewsym. matrix

    ss_matrix = torch.zeros((3, 3), device=vector.device)
    ss_matrix[0, 1] = -vector[2]
    ss_matrix[0, 2] = vector[1]
    ss_matrix[1, 0] = vector[2]
    ss_matrix[1, 2] = -vector[0]
    ss_matrix[2, 0] = -vector[1]
    ss_matrix[2, 1] = vector[0]

    return ss_matrix


class OptSO3T(nn.Module):
    def __init__(self):
        super(OptSO3T, self).__init__()
        # S = [self.v, self.w] represents the screw axis; θ the magnitude
        # self.v = I + (1 - cosθ)/θ**2
        self.w = nn.Parameter(torch.normal(0.0, 1e-6, size=(3,)))  # θ
        self.t = nn.Parameter(torch.normal(0.0, 1e-6, size=(3,)))  # ρ
        self.theta = nn.Parameter(torch.normal(0.0, 1e-6, size=()))  #

    def forward(self, x):
        T_i = x.clone()
        w_skewsym = vec2ss_matrix(self.w)  # [w]_x -> the 3 x 3 skew-sym matrix of w
        R_exp = (
            torch.eye(3, device=x.device)
            + torch.sin(self.theta) * w_skewsym
            + (1 - torch.cos(self.theta)) * torch.matmul(w_skewsym, w_skewsym)
        )
        T_i[:, :3, :3] = torch.matmul(x[:, :3, :3], R_exp)
        T_i[:, :3, 3] += self.t
        return T_i


class OptSE3(nn.Module):
    def __init__(self):
        super(OptSE3, self).__init__()
        self.w = nn.Parameter(torch.normal(0.0, 1e-6, size=(3,)))
        self.v = nn.Parameter(torch.normal(0.0, 1e-6, size=(3,)))
        self.theta = nn.Parameter(torch.normal(0.0, 1e-6, size=()))

    def forward(self, x):
        exp_i = torch.zeros((4, 4), device=x.device)
        w_skewsym = vec2ss_matrix(self.w)
        exp_i[:3, :3] = (
            torch.eye(3, device=x.device)
            + torch.sin(self.theta) * w_skewsym
            + (1 - torch.cos(self.theta)) * torch.matmul(w_skewsym, w_skewsym)
        )
        exp_i[:3, 3] = torch.matmul(
            torch.eye(3, device=x.device) * self.theta
            + (1 - torch.cos(self.theta)) * w_skewsym
            + (self.theta - torch.sin(self.theta)) * torch.matmul(w_skewsym, w_skewsym),
            self.v,
        )
        exp_i[3, 3] = 1.0
        T_i = torch.matmul(exp_i, x)
        return T_i
