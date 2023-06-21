import numpy as np
import scipy

class linear_cca():
    def __init__(self):
        self.w = [None, None]
        self.m = [None, None]

    def fit(self, H1, H2, outdim_size):

        r1 = 1e-4 + 1j*1e-6
        r2 = 1e-4 + 1j*1e-6

        m = H1.shape[0]
        o1 = H1.shape[1]
        o2 = H2.shape[1]

        self.m[0] = np.mean(H1, axis=0)
        self.m[1] = np.mean(H2, axis=0)
        H1bar = H1 - np.tile(self.m[0], (m, 1))
        H2bar = H2 - np.tile(self.m[1], (m, 1))

        SigmaHat12 = (1.0 / (m - 1)) * np.dot(H1bar.conjugate().T, H2bar)
        SigmaHat11 = (1.0 / (m - 1)) * np.dot(H1bar.conjugate().T, H1bar) + r1 * np.identity(o1)
        SigmaHat22 = (1.0 / (m - 1)) * np.dot(H2bar.conjugate().T, H2bar) + r2 * np.identity(o2)

        SigmaTilde12 = (1.0 / (m - 1)) * np.dot(H1bar.T, H2bar)
        SigmaTilde11 = (1.0 / (m - 1)) * np.dot(H1bar.T, H1bar) + r1 * np.identity(o1)
        SigmaTilde22 = (1.0 / (m - 1)) * np.dot(H2bar.T, H2bar) + r2 * np.identity(o2)

        AugmentedSigma12 = np.concatenate((np.concatenate((SigmaHat12, SigmaTilde12), axis=1),
                                           np.concatenate((SigmaTilde12.conjugate(), SigmaHat12.conjugate()), axis=1)), axis = 0)
        AugmentedSigma11 = np.concatenate((np.concatenate((SigmaHat11, SigmaTilde11), axis=1),
                                           np.concatenate((SigmaTilde11.conjugate(), SigmaHat11.conjugate()), axis=1)), axis = 0)
        AugmentedSigma22 = np.concatenate((np.concatenate((SigmaHat22, SigmaTilde22), axis=1),
                                           np.concatenate((SigmaTilde22.conjugate(), SigmaHat22.conjugate()), axis=1)), axis = 0)

        [D1, V1] = scipy.linalg.eig(AugmentedSigma11)
        [D2, V2] = scipy.linalg.eig(AugmentedSigma22)

        AugSigma11RootInv = np.dot(np.dot(V1, np.diag(D1 ** -0.5)), V1.conjugate().T)
        AugSigma22RootInv = np.dot(np.dot(V2, np.diag(D2 ** -0.5)), V2.conjugate().T)

        Tval = np.dot(np.dot(AugSigma11RootInv, AugmentedSigma12), AugSigma22RootInv)

        C = (1/np.sqrt(2)) * np.concatenate((np.concatenate((np.identity(o1), 1j*np.identity(o1)), axis=1),
                                             np.concatenate((np.identity(o1), -1j*np.identity(o1)), axis=1)), axis = 0)

        T_hat = np.dot(np.dot(C.conjugate().T, Tval), C)
        T_hat = T_hat.real

        [U_hat, K, V_hat] = scipy.linalg.svd(T_hat)
        V_hat = V_hat.conjugate().T

        U = np.dot(np.dot(C, U_hat), C.conjugate().T)
        V = np.dot(np.dot(C, V_hat), C.conjugate().T)

        A = np.dot(U.conjugate().T, AugSigma11RootInv)
        B = np.dot(V.conjugate().T, AugSigma22RootInv)

        self.w[0] = [A[0:outdim_size, 0:outdim_size], A[0:outdim_size, outdim_size:A.shape[1]]]
        self.w[1] = [B[0:outdim_size, 0:outdim_size], B[0:outdim_size, outdim_size:A.shape[1]]]

    def _get_result(self, x, idx):
        input = x - self.m[idx].reshape([1, -1]).repeat(len(x), axis=0)
        input_conj = input.conjugate()
        result = np.dot(input, self.w[idx][0]) + np.dot(input_conj, self.w[idx][1])
        return result

    def transform(self, H1, H2):
        return self._get_result(H1, 0), self._get_result(H2, 1)
