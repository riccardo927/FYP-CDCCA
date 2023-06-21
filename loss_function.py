import torch
import numpy as np

class cdcca_loss():
    def __init__(self, outdim_size, device):
        self.outdim_size = outdim_size
        self.device = device

    def loss(self, H1, H2):

        r1 = 1e-3 + 1j*1e-8
        r2 = 1e-3 + 1j*1e-8

        H1, H2 = H1.t(), H2.t()

        o1 = o2 = H1.size(0)
        m = H1.size(1)

        H1bar = H1 - H1.mean(dim=1).unsqueeze(dim=1)
        H2bar = H2 - H2.mean(dim=1).unsqueeze(dim=1)

        # Compute covariance matrices and add regularization term so they are positive definite
        SigmaHat12 = (1.0 / (m - 1)) * torch.matmul(H1bar, H2bar.conj().t())
        SigmaHat11 = (1.0 / (m - 1)) * torch.matmul(H1bar, H1bar.conj().t()) + r1 * torch.eye(o1, device=self.device)
        SigmaHat22 = (1.0 / (m - 1)) * torch.matmul(H2bar, H2bar.conj().t()) + r2 * torch.eye(o2, device=self.device)

        SigmaTilde12 = (1.0 / (m - 1)) * torch.matmul(H1bar, H2bar.t())
        SigmaTilde11 = (1.0 / (m - 1)) * torch.matmul(H1bar, H1bar.t()) + r1 * torch.eye(o1, device=self.device)
        SigmaTilde22 = (1.0 / (m - 1)) * torch.matmul(H2bar, H2bar.t()) + r2 * torch.eye(o2, device=self.device)

        AugmentedSigma12 = torch.cat((torch.cat((SigmaHat12, SigmaTilde12), 1),torch.cat((torch.conj(SigmaTilde12), torch.conj(SigmaHat12)), 1)), 0)
        AugmentedSigma11 = torch.cat((torch.cat((SigmaHat11, SigmaTilde11), 1),torch.cat((torch.conj(SigmaTilde11), torch.conj(SigmaHat11)), 1)), 0)
        AugmentedSigma22 = torch.cat((torch.cat((SigmaHat22, SigmaTilde22), 1),torch.cat((torch.conj(SigmaTilde22), torch.conj(SigmaHat22)), 1)), 0)

        [D1, V1] = torch.linalg.eig(AugmentedSigma11)
        [D2, V2] = torch.linalg.eig(AugmentedSigma22)

        AugSigma11RootInv = torch.matmul(torch.matmul(V1, torch.diag(D1 ** -0.5)), V1.conj().t())
        AugSigma22RootInv = torch.matmul(torch.matmul(V2, torch.diag(D2 ** -0.5)), V2.conj().t())

        Tval = torch.matmul(torch.matmul(AugSigma11RootInv, AugmentedSigma12), AugSigma22RootInv)

        C = (1/np.sqrt(2)) * torch.cat((torch.cat((torch.eye(o1, device=self.device), torch.complex(torch.zeros(o1, o1, device = self.device), torch.eye(o1, device=self.device))), 1),
                                        torch.cat((torch.eye(o1, device=self.device), torch.complex(torch.zeros(o1, o1, device = self.device), -torch.eye(o1, device=self.device))), 1)), 0)
        
        T_hat = torch.matmul(torch.matmul(C.conj().t(), Tval), C)
        T_hat = T_hat.real

        [_, K, _] = torch.linalg.svd(T_hat)

        corr = 1/(o1*2) * torch.sum(K)

        return - corr.real