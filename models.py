# x0 = 0 - Վեկտոր
# A, B - Մատրիցի գործակիցներ են
# Q, R - Օպտիմալացման ցուցանիշի պարամետրեր
# z0 - Վրդովմունքներ, խանգարումների ցուցանիշ
# Tz - Ժամանակի պարամետր
# dT - Քայլ
# k(t) - Ռիկատիի հավասարումներ
# k1(t) - Ռիկատիի հավասարումներ

import math
import numpy
from scipy.integrate import quad
import matplotlib.pyplot as plt


class DecompositionMethod:
    x0 = 0
    z0 = 1
    t0 = 1
    t = 10
    dT = 1
    Tz = 4
    _k = {10: 0.41, 9: 0.41, 8: 0.41, 7: 0.41, 6: 0.41, 5: 0.41, 4: 0.41, 3: 0.41, 2: 0.4, 1: 0.39}
    _k1 = {10: 0.25, 9: 0.25, 8: 0.25, 7: 0.25, 6: 0.25, 5: 0.25, 4: 0.25, 3: 0.24, 2: 0.125, 1: 0.15}
    tf = tuple(range(t, t0-1, -dT))

    def __init__(self, q, r, a, b):
        self.Q = q
        self.R = r
        self.A = a
        self.B = b

        # self.tf = [self.t]
        # while self.tf[-1] > self.t0:
        #     self.tf.append(self.tf[-1] - self.dT)

    def z1(self, t: float):
        return self.z0 * numpy.e ** (t / self.Tz)

    def k(self, t) -> float:
        return self._k.get(t)

    def k1(self, t) -> float:
        return self._k1.get(t)

    def U(self, t: float):
        return (-self.R) ** (-1) * self.B * (self.k(t) * self.x0 + self.k1(t))

    def X(self, t) -> float:
        return self.A*self.x0 + self.B*self.U(t) + self.z1(t)

    def I(self, _tf: float):
        return quad(lambda t: self.X(_tf) * self.Q * self.X(_tf) + self.U(_tf) * self.R * self.U(_tf), self.t0, _tf)


def run():
    Q = 1
    R = 1
    A = -1
    B = 1

    de_method = DecompositionMethod(q=Q, r=R, a=A, b=B)

    U = [de_method.U(t) for t in de_method.tf]
    I = [de_method.I(t)[0] for t in de_method.tf]
    X = [de_method.X(t) for t in de_method.tf]

    plt.figure()
    plt.subplot(221)
    plt.plot(de_method.tf, de_method._k1.values(), 'r-*')
    plt.ylabel('k1')
    plt.grid(True)

    plt.subplot(223)
    plt.plot(de_method.tf, U, 'g-*')
    plt.ylabel('U')
    plt.grid(True)

    plt.subplot(222)
    plt.plot(de_method.tf, I, 'b-*')
    plt.ylabel('I')
    plt.grid(True)

    plt.subplot(224)
    plt.plot(de_method.tf, X, 'y-*')
    plt.ylabel('X')
    plt.grid(True)
    plt.show()



if __name__ == '__main__':
    run()
