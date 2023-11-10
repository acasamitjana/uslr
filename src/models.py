import itertools
import pdb
import copy

import numpy as np
import torch
from torch import nn

#########################################
##   Linear Registration/Deformation   ##
#########################################

class InstanceRigidModel(nn.Module):

    def __init__(self, timepoints, reg_weight=0.001, cost='l1', device='cpu', torch_dtype=torch.float):
        super().__init__()

        self.device = device
        self.cost = cost
        self.reg_weight = reg_weight

        self.timepoints = timepoints
        self.N = len(timepoints)
        self.K = int(self.N * (self.N-1) / 2)

        # Parameters
        self.angle = torch.nn.Parameter(torch.zeros(3, self.N))
        self.translation = torch.nn.Parameter(torch.zeros(3, self.N))
        self.angle.requires_grad = True
        self.translation.requires_grad = True


    def _compute_matrix(self):

        angles = self.angle / 180 * np.pi

        cos = torch.cos(angles)
        sin = torch.sin(angles)

        T = torch.zeros((4, 4, self.N))
        T[0, 0] = cos[2]*cos[1]
        T[1, 0] = sin[2]*cos[1]
        T[2, 0] = -sin[1]

        T[0, 1] = cos[2] * sin[1] * sin[0] - sin[2] * cos[0]
        T[1, 1] = sin[2] * sin[1] * sin[0] + cos[2] * cos[0]
        T[2, 1] = cos[1] * sin[0]

        T[0, 2] = cos[2] * sin[1] * cos[0] + sin[2] * sin[0]
        T[1, 2] = sin[2] * sin[1] * cos[0] - cos[2] * sin[0]
        T[2, 2] = cos[1] * cos[0]

        T[0, 3] = self.translation[0]# + self.tr0[0]
        T[1, 3] = self.translation[1]# + self.tr0[1]
        T[2, 3] = self.translation[2]# + self.tr0[2]
        T[3, 3] = 1

        #
        # for n in range(self.N):
        #
        #     T[..., n] = torch.chain_matmul(self.T0inv, T[..., n], self.T0)

        return T


    def _build_combinations(self, timepoints, latent_matrix):

        K = self.K
        if any([isinstance(t, str) for t in timepoints]):
            timepoints_dict = {
                t: it_t for it_t, t in enumerate(timepoints)
            }
        else:
            timepoints_dict = {
                t.id: it_t for it_t, t in enumerate(timepoints)
            }  # needed for non consecutive timepoints (if we'd like to skip one for whatever reason)


        Tij = torch.zeros((4, 4, K))

        k = 0
        for tp_ref, tp_flo in itertools.combinations(timepoints, 2):

            if not isinstance(tp_ref, str):
                t0 = timepoints_dict[tp_ref.id]
                t1 = timepoints_dict[tp_flo.id]
            else:
                t0 = timepoints_dict[tp_ref]
                t1 = timepoints_dict[tp_flo]


            T0k = latent_matrix[..., t0]
            T1k = latent_matrix[..., t1]

            Tij[..., k] = torch.matmul(T1k, torch.inverse(T0k))

            k += 1

        return Tij


    def _compute_log(self, Tij):

        K = Tij.shape[-1]
        R = Tij[:3, :3]
        Tr = Tij[:3, 3]

        logTij = torch.zeros((6, K))

        eps = 1e-6
        for k in range(K):
            t_norm = torch.arccos(torch.clamp((torch.trace(R[..., k]) - 1) / 2, min=-1+eps, max=1-eps)) + eps
            W = 1 / (2 * torch.sin(t_norm)) * (R[..., k] - R[..., k].T) * t_norm
            Vinv = torch.eye(3) - 0.5 * W + ((1 - (t_norm * torch.cos(t_norm / 2)) / (2 * torch.sin(t_norm / 2))) / t_norm ** 2) * W*W#torch.matmul(W, W)


            logTij[0, k] = 1 / (2 * torch.sin(t_norm)) * (R[..., k][2, 1] - R[..., k][1, 2]) * t_norm
            logTij[1, k] = 1 / (2 * torch.sin(t_norm)) * (R[..., k][0, 2] - R[..., k][2, 0]) * t_norm
            logTij[2, k] = 1 / (2 * torch.sin(t_norm)) * (R[..., k][1, 0] - R[..., k][0, 1]) * t_norm

            logTij[3:,k] = torch.matmul(Vinv, Tr[..., k])

        return logTij


    def forward(self, logRobs, timepoints):
        Ti = self._compute_matrix()
        Tij = self._build_combinations(timepoints, Ti)
        logTij = self._compute_log(Tij)
        logTi = self._compute_log(Ti)

        if self.cost == 'l1':
            loss = torch.sum(torch.sqrt(torch.sum((logTij - logRobs) ** 2, axis=0))) / self.K
        elif self.cost == 'l2':
            loss = torch.sum((logTij - logRobs) ** 2 + 1e-6) / self.K
        else:
            raise ValueError('Cost ' + self.cost + ' not valid. Choose \'l1\' of \'l2\'.' )
        loss += self.reg_weight * torch.sum(logTi**2) / self.K
        return loss

class InstanceRigidModelLOG(nn.Module):

    def __init__(self, timepoints, reg_weight=0.001, cost='l1', device='cpu', torch_dtype=torch.float):
        super().__init__()

        self.device = device
        self.cost = cost
        self.reg_weight = reg_weight

        self.timepoints = timepoints
        self.N = len(timepoints)
        self.K = int(self.N * (self.N-1) / 2)

        # Parameters
        self.angle = torch.nn.Parameter(torch.zeros(3, self.N))
        self.translation = torch.nn.Parameter(torch.zeros(3, self.N))
        self.angle.requires_grad = True
        self.translation.requires_grad = True


    def _compute_matrix(self):

        T = torch.zeros((4,4,self.N))
        for n in range(self.N):
            theta = torch.sqrt(torch.sum(self.angle[..., n]**2)) # torch.sum(torch.abs(self.angle))
            W = torch.zeros((3,3))
            W[1,0], W[0,1] = self.angle[2, n], -self.angle[2, n]
            W[0,2], W[2,0] = self.angle[1, n], -self.angle[1, n]
            W[2,1], W[1,2] = self.angle[0, n], -self.angle[0, n]
            V = torch.eye(3) + (1 - torch.cos(theta)) / (theta ** 2) * W + (theta - torch.sin(theta)) / (theta ** 3) * torch.matmul(W,W)

            T[:3, :3, n] = torch.eye(3) + torch.sin(theta) / theta * W      +      (1 - torch.cos(theta)) / (theta ** 2) * torch.matmul(W,W)
            T[:3, 3, n] = V @ self.translation[..., n]#torch.matmul(V, self.translation[..., n])
            T[3, 3, n] = 1

            #
            # for n in range(self.N):
            #
            #     T[..., n] = torch.chain_matmul(self.T0inv, T[..., n], self.T0)

        return T


    def _build_combinations(self, timepoints):

        K = self.K
        if any([isinstance(t, str) for t in timepoints]):
            timepoints_dict = {
                t: it_t for it_t, t in enumerate(timepoints)
            }
        else:
            timepoints_dict = {
                t.id: it_t for it_t, t in enumerate(timepoints)
            }  # needed for non consecutive timepoints (if we'd like to skip one for whatever reason)

        Tij = torch.zeros((6, K))

        k = 0
        for tp_ref, tp_flo in itertools.combinations(timepoints, 2):

            if not isinstance(tp_ref, str):
                t0 = timepoints_dict[tp_ref.id]
                t1 = timepoints_dict[tp_flo.id]
            else:
                t0 = timepoints_dict[tp_ref]
                t1 = timepoints_dict[tp_flo]
            Tij[:3, k] = self.angle[..., t1] - self.angle[..., t0]
            Tij[3:, k] = self.translation[..., t1] - self.translation[..., t0]

            k += 1

        return Tij



    def forward(self, logRobs, timepoints):

        logTij = self._build_combinations(timepoints)
        if self.cost == 'l1':
            loss = torch.sum(torch.sqrt(torch.sum((logTij - logRobs) ** 2, axis=0))) / self.K
        elif self.cost == 'l2':
            loss = torch.sum((logTij - logRobs) ** 2 + 1e-6) / self.K
        else:
            raise ValueError('Cost ' + self.cost + ' not valid. Choose \'l1\' of \'l2\'.' )
        loss += self.reg_weight * torch.sum(torch.sum(self.angle**2, axis=0) + torch.sum(self.translation**2, axis=0), axis=0) # / self.K

        return loss

###################
##   Functions   ##
###################



