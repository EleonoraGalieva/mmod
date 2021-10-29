import numpy as np
import math
import scipy.stats as sts
from collections import Counter
import matplotlib.pyplot as plt

fig, axs = plt.subplots(3, 2)


# Алгоритм получения значений системы дискретных случайных величин
def form_discrete_system(probability_matrix, A, B):
    n, m = probability_matrix.shape
    # Вычисляем суммы qi и lk:
    q = np.sum(probability_matrix, axis=1)
    l = np.zeros(n)
    l[0] = q[0]
    for i in range(1, n):
        l[i] = l[i - 1] + q[i]
    # Находим индекс первой компоненты дискретной величины
    first_index = np.searchsorted(l, np.random.rand())
    # Находим значение первой компоненты дискретной величины
    x1 = A[first_index]
    # Вычисляем сумму rs
    r = np.zeros(m)
    r[0] = probability_matrix[first_index, 0]
    for i in range(1, m):
        r[i] = r[i - 1] + probability_matrix[first_index, i]
    # Находим значение второй компоненты дискретной величины
    second_index = np.searchsorted(r, np.random.rand() * r[-1])
    x2 = B[second_index]
    return x1, x2


############### Статистические исследования полученных величин: ####################

# Эмпирическая матрица вероятностей:
def find_empiric_matrix(n, A, B, probability_matrix):
    empiric_matrix = np.zeros(probability_matrix.shape)
    discrete_values_array = Counter([form_discrete_system(probability_matrix, A, B) for i in range(n)])
    for (x1, x2), counter in discrete_values_array.items():
        first_index = list(A).index(x1)
        second_index = list(B).index(x2)
        empiric_matrix[first_index, second_index] = counter / n
    return empiric_matrix


def draw_histogram(A, B, empiric_matrix):
    x1_probability = np.sum(empiric_matrix, axis=1)
    x2_probability = np.sum(empiric_matrix, axis=0)
    axs[0][0].bar(A, x1_probability, color=(0.2, 0.4, 0.6, 0.6))
    axs[0][1].bar(B, x2_probability, color=(0.3, 0.6, 0.6, 0.6))
    axs[0][0].legend(['X1'])
    axs[0][1].legend(['X2'])


def point_estimate_M(X, n):
    return math.fsum(X) / n


def point_estimate_D(X, n, M_estimate):
    return 1 / (n - 1) * math.fsum(list(map(lambda xi: (xi - M_estimate) ** 2, X)))


def intervals_for_M(n, D_estimate, M_estimate, i):
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
    axs[1][i].plot([0.9, 0.94, 0.98, 0.99], [2 * delta1, 2 * delta2, 2 * delta3, 2 * delta4])
    return [interval_for_m_1, interval_for_m_2, interval_for_m_3, interval_for_m_4]


def intervals_for_D(n, d, m, i):
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
    axs[2][i].plot([0.95, 0.98, 0.99],
                   [interval_for_d_1[1] - interval_for_d_1[0], interval_for_d_2[1] - interval_for_d_2[0],
                    interval_for_d_3[1] - interval_for_d_3[0]])
    return [interval_for_d_1, interval_for_d_2, interval_for_d_3]


def find_correlation(M_x1, M_x2, D_x1, D_x2, empiric_matrix, A, B):
    M_x1_x2 = 0
    for i in range(empiric_matrix.shape[0]):
        for j in range(empiric_matrix.shape[1]):
            M_x1_x2 += A[i] * B[j] * empiric_matrix[i, j]
    return (M_x1_x2 - M_x1 * M_x2) / math.sqrt(D_x1) * math.sqrt(D_x2)


def pearson_criterion(theoretical_matrix, empiric_matrix, n):
    chi2 = n * np.sum((empiric_matrix - theoretical_matrix) ** 2 / theoretical_matrix)
    chi2_value = sts.chi2.ppf(0.95, theoretical_matrix.size - 1)
    return chi2 < chi2_value


if __name__ == '__main__':
    P = np.array([[0.2, 0.3], [0.1, 0.2], [0.1, 0.1]])
    A = np.array([1, 2, 4])
    B = np.array([1, 3])
    print('Теоретическая матрица:')
    print(P)
    print('Эмпирическая матрица (для n = 10):')
    print(find_empiric_matrix(10, A, B, P))
    print('Эмпирическая матрица (для n = 100):')
    empiric_matrix = find_empiric_matrix(100, A, B, P)
    print(empiric_matrix)
    print('Эмпирическая матрица (для n = 1000):')
    print(find_empiric_matrix(1000, A, B, P))
    draw_histogram(A, B, empiric_matrix)

    n = 100
    X = [form_discrete_system(P, A, B) for i in range(n)]
    x1 = [X[i][0] for i in range(n)]
    x2 = [X[i][1] for i in range(n)]
    print('Точечные оценки:')
    print('Точечные оценки мат. ожидания:')
    M_x1 = point_estimate_M(x1, n)
    M_x2 = point_estimate_M(x2, n)
    print('Для x1: ' + str(M_x1))
    print('Для x2: ' + str(M_x2))
    print('Точечные оценки дисперсии:')
    D_x1 = point_estimate_D(x1, n, M_x1)
    D_x2 = point_estimate_D(x2, n, M_x2)
    print('Для x1: ' + str(D_x1))
    print('Для x2: ' + str(D_x2))
    print('Интервальные оценки:')
    print('Интервальные оценки для мат. ожидания:')
    print('Для x1:')
    intervals_for_M(n, D_x1, M_x1, 0)
    print('Для x2:')
    intervals_for_M(n, D_x1, M_x2, 1)
    print('Интервальные оценки для дисперсии:')
    print('Для x1:')
    intervals_for_D(n, D_x1, M_x1, 0)
    print('Для x2:')
    intervals_for_D(n, D_x1, M_x2, 1)
    print('Коэффициент корреляции:')
    print(find_correlation(M_x1, M_x1, D_x1, D_x2, empiric_matrix, A, B))
    if pearson_criterion(P, empiric_matrix, n):
        print('Фактические данные не противоречат ожидаемым по критерию Пирсона.')
    plt.show()
