from math import inf
import numpy as np
import simpy

from lab2 import QueuingSystemModel, find_empiric_probabilities, run_model


def find_theoretical_probabilities(applications_flow_rate, service_flow_rate):
    print('-------------------Theoretical---------------------')
    ro = applications_flow_rate / service_flow_rate
    p0 = 1 - ro
    print('P0: ' + str(p0))
    for k in range(1, 6):
        pk = ro ** k * p0
        print('P' + str(k) + ': ' + str(pk))
    # Because queue has no max length
    P_rej = 0
    print('Theoretical probability of rejection: ' + str(P_rej))
    Q = 1
    print('Theoretical Q: ', Q)
    A = applications_flow_rate * Q
    print('Theoretical A: ', A)
    L_q = ro ** 2 / p0
    print('Average queue length is: ', L_q)
    L_pr = ro*Q
    print('Average amount of applications in QS (both processing and waiting): ', L_pr + L_q)
    print('Average time in queue is: ', L_q / A)
    print('Average amount of busy channels: ', Q * ro)
    print('Average time in QS is: ',  (L_pr + L_q) / A)


if __name__ == '__main__':
    # 3.	Железнодорожная сортировочная горка, на которую по дается простейший поток составов с интенсивностью X = 2
    # состава в час, представляет собой одноканальную СМО с неограниченной очередью. Время обслуживания (роспуска)
    # состава на горке имеет показательное распределение со средним значением tобсл = 20 мин. Найти финальные
    # вероятности состояний СМО, среднее число z составов, связанных с горкой, среднее число составов в очереди,
    # среднее время tCOCT пребывания состава в СМО, среднее время tQ пребывания состава в очереди.

    channels_number = 1
    print('Amount of channels (n): ' + str(channels_number))
    service_flow_rate = 3
    print('Service flow rate (mu): ' + str(service_flow_rate))
    applications_flow_rate = 2
    print('Applications flow rate (lambda): ' + str(applications_flow_rate))
    env = simpy.Environment()
    model = QueuingSystemModel(env, channels_number, service_flow_rate, applications_flow_rate)
    print('Running simulation...')
    env.process(run_model(env, model))
    # How long the simulation will run
    env.run(until=300)

    find_empiric_probabilities(np.array(model.applications_done), np.array(model.applications_rejected),
                               np.array(model.queue_times),
                               np.array(model.total_wait_times),
                               np.array(model.total_qs_list),
                               np.array(model.queue_list), channels_number, 5, applications_flow_rate,
                               service_flow_rate)
    find_theoretical_probabilities(applications_flow_rate, service_flow_rate)
