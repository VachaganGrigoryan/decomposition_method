# x0 = 0 - Վեկտոր
# A, B - Մատրիցի գործակիցներ են
# Q, R - Օպտիմալացման ցուցանիշի պարամետրեր
# z0 - Վրդովմունքներ, խանգարումների ցուցանիշ
# Tz - Ժամանակի պարամետր
# dT - Քայլ
# k(t) - Ռիկատիի հավասարումներ
# k1(t) - Ռիկատիի հավասարումներ

import numpy
from scipy.integrate import quad


class DecompositionMethod:
    x0 = 0
    z0 = 1
    t0 = 1
    t = 10
    dT = 0.2
    Tz = 4

    def __init__(self, q, r, b):
        self.Q = q
        self.R = r
        self.B = b

        self.tf = [self.t]
        while self.tf[-1] > self.t0:
            self.tf.append(self.tf[-1] - self.dT)

    def z1(self, t: float):
        return self.z0 * numpy.e ** (t / self.Tz)

    def k(self, t) -> float:
        return 0.41

    def k1(self, t) -> float:
        return 0.25

    def U(self, t: float):
        return -self.R ** -1 * self.B * (self.k(t) * self.x0 + self.k1(t))

    def A(self, t):
        return -1

    def X(self, t) -> float:
        return self.A(t) + self.B*self.U(t) + self.z1(t)

    def I(self, _tf: float):
        return quad(lambda t: self.X(t) * self.Q * self.X(t) + self.U(t) * self.R * self.U(t), self.t0, _tf)


def run():
    Q = 1
    R = 1
    B = 1

    de_method = DecompositionMethod(q=Q, r=R, b=B)

    for t in de_method.tf:
        print(de_method.I(t))


if __name__ == '__main__':
    run()




