import math
import scipy.stats as sts
import matplotlib.pyplot as plt
from scipy import optimize

# Пусть совместная плотность распределения:
# f(x,y) = e^(x+y), где
# 1<=y<=2
# 2<=x<=3

# Исп. в формировании БСВ
A = [10000]


def f(x, y):
    return math.exp(x + y)


# Найдем частную функцию плотности f(x) (через интеграл совместной плотности распределения по переменной y)
def f_x(x):
    return math.exp(x + 2) - math.exp(x + 1)


# Найдем вторую частную функцию плотности f(y) на основании теоремы умножения плотностей распределения
def f_y(y):
    return math.exp(y) / (math.exp(2) - math.exp(1))


# Используем метод Неймана для получения значения СВ в соответствии с законом распределения
def Neumanns_method(f, a, b):
    # Находим максимум f
    W = -optimize.minimize_scalar(lambda t: -f(t), bounds=[a, b], method='bounded')['fun']
    while True:
        # С помощью датчика случайных чисел, получаем пару чисел, равномерно распределенных на (0,1)
        z = multiplicative_congruent_method()
        x1_ = a + z[0] * (b - a)
        x2_ = z[1] * W
        if f(x1_) < x2_:
            return x1_


# Используем мультипликативный конгруэнтный датчик для формирования БСВ
#  Подобранные коэф. соответствуют Apple CarbonLib, C++11's minstd_rand0
def multiplicative_congruent_method(m=2 ** 31 - 1, k=16807, amount=2):
    z = []
    for i in range(amount):
        A.append((k * A[len(A) - 1]) % m)
        z.append(A[len(A) - 1] / m)
    return z


def analytical_transformation_method(n):
    Z = []
    Z1 = []
    Z2 = []
    while n:
        z1 = Neumanns_method(f_x, 2, 3)
        Z1.append(z1)
        z2 = Neumanns_method(f_y, 1, 2)
        Z2.append(z2)
        Z.append((z1, z2))
        n -= 1
    return Z, Z1, Z2


############### Статистические исследования полученных величин: ####################

def draw_histograms(z):
    plt.hist(z, density=True, bins=30)
    plt.xlabel('Data')
    plt.ylabel('Probability')
    plt.show()


def point_estimate_M(X, n):
    return math.fsum(X) / n


def point_estimate_D(X, n, M_estimate):
    return 1 / (n - 1) * math.fsum(list(map(lambda xi: (xi - M_estimate) ** 2, X)))


def intervals_for_M(n, D_estimate, M_estimate):
    tt = sts.t(n)
    arr = tt.rvs(1000000)
    delta1 = sts.mstats.mquantiles(arr, prob=0.9) * math.sqrt(D_estimate / (n - 1))
    delta2 = sts.mstats.mquantiles(arr, prob=0.94) * math.sqrt(D_estimate / (n - 1))
    delta3 = sts.mstats.mquantiles(arr, prob=0.98) * math.sqrt(D_estimate / (n - 1))
    delta4 = sts.mstats.mquantiles(arr, prob=0.99) * math.sqrt(D_estimate / (n - 1))
    interval_for_m_1 = [M_estimate - delta1[0], M_estimate + delta1[0]]
    print("Доверительный интервал для мат. ожидания при доверительной вероятности 0.9 и n = " + str(n) + ": " + str(
        interval_for_m_1))
    interval_for_m_2 = [M_estimate - delta2[0], M_estimate + delta2[0]]
    print(
        "Доверительный интервал для мат. ожидания при доверительной вероятности 0.94 и n = " + str(n) + ": " + str(
            interval_for_m_2))
    interval_for_m_3 = [M_estimate - delta3[0], M_estimate + delta3[0]]
    print(
        "Доверительный интервал для мат. ожидания при доверительной вероятности 0.98 и n = " + str(n) + ": " + str(
            interval_for_m_3))
    interval_for_m_4 = [M_estimate - delta4[0], M_estimate + delta4[0]]
    print(
        "Доверительный интервал для мат. ожидания при доверительной вероятности 0.99 и n = " + str(n) + ": " + str(
            interval_for_m_4))
    # m+delta-m+delta=2delta
    plt.plot([0.9, 0.94, 0.98, 0.99], [2 * delta1, 2 * delta2, 2 * delta3, 2 * delta4])
    plt.show()
    return [interval_for_m_1, interval_for_m_2, interval_for_m_3, interval_for_m_4]


def intervals_for_D(n, d, m):
    CHI2 = sts.chi2(n - 1)
    arr = CHI2.rvs(1000000)
    delta1 = sts.mstats.mquantiles(arr, prob=[0.025, 0.975])
    delta2 = sts.mstats.mquantiles(arr, prob=[0.01, 0.99])
    delta3 = sts.mstats.mquantiles(arr, prob=[0.005, 0.995])
    interval_for_d_1 = [d - (n - 1) * d / delta1[1], d + (n - 1) * d / delta1[0]]
    print("Доверительный интервал для дисперсии при доверительной вероятности 0.9 и n = " + str(n) + ": " + str(
        interval_for_d_1))
    interval_for_d_2 = [d - (n - 1) * d / delta2[1], d + (n - 1) * d / delta2[0]]
    print(
        "Доверительный интервал для дисперсии при доверительной вероятности 0.94 и n = " + str(n) + ": " + str(
            interval_for_d_2))
    interval_for_d_3 = [d - (n - 1) * d / delta3[1], d + (n - 1) * d / delta3[0]]
    print(
        "Доверительный интервал для дисперсии при доверительной вероятности 0.98 и n = " + str(n) + ": " + str(
            interval_for_d_3))
    # m+delta-m+delta=2delta
    plt.plot([0.95, 0.98, 0.99],
             [interval_for_d_1[1] - interval_for_d_1[0], interval_for_d_2[1] - interval_for_d_2[0],
              interval_for_d_3[1] - interval_for_d_3[0]])
    plt.show()
    return [interval_for_d_1, interval_for_d_2, interval_for_d_3]


if __name__ == '__main__':
    n = 1000
    z, z1, z2 = analytical_transformation_method(n)
    print('Полученные двумерные СВ для n = ' + str(n))
    print(z)
    draw_histograms(z1)
    draw_histograms(z2)
    print('Точечные оценки:')
    print('Точечные оценки мат. ожидания:')
    M_y = point_estimate_M(z1, n)
    M_x = point_estimate_M(z2, n)
    print('Для x: ' + str(M_x))
    print('Для y: ' + str(M_y))
    print('Точечные оценки дисперсии:')
    D_y = point_estimate_D(z1, n, M_y)
    D_x = point_estimate_D(z2, n, M_x)
    print('Для x: ' + str(D_x))
    print('Для y: ' + str(D_y))
    print('Интервальные оценки:')
    print('Интервальные оценки для мат. ожидания:')
    print('Для x:')
    intervals_for_M(n, D_x, M_x)
    print('Для y:')
    intervals_for_M(n, D_y, M_y)
    print('Интервальные оценки для дисперсии:')
    print('Для x:')
    intervals_for_D(n, D_x, M_x)
    print('Для y:')
    intervals_for_D(n, D_y, M_y)
