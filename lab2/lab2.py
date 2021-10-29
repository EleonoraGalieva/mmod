from math import factorial
import simpy
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

# channels_number - n
# service_flow_rate - mu
# applications_flow_rate - lambda
# max_queue_length - m
# queue_waiting_flow_rate - v

fig, axs = plt.subplots(2)


class QueuingSystemModel:
    def __init__(self, env, channels_number, service_flow_rate, applications_flow_rate, queue_waiting_flow_rate,
                 max_queue_length):
        self.env = env
        self.service_flow_rate = service_flow_rate
        self.applications_flow_rate = applications_flow_rate
        self.queue_waiting_flow_rate = queue_waiting_flow_rate
        self.max_queue_length = max_queue_length

        # How long each application is in QS
        self.total_wait_times = []
        # How many applications are in QS at the moment (both in queue and processing)
        self.total_qs_list = []
        # How long each application is in queue
        self.queue_times = []
        # How many applications are in queue at the moment
        self.queue_list = []
        # How many applications have been processed at moment of time
        self.applications_done = []
        # How many applications have been rejected at moment of time
        self.applications_rejected = []

        self.channel = simpy.Resource(env, channels_number)

    def application_processing(self, application):
        # print('Application ' + str(application) + ' is processing.')
        # Trigger an event after a certain amount of time has passed
        yield self.env.timeout(np.random.exponential(1 / self.service_flow_rate))

    def application_waiting(self, application):
        # print('Application ' + str(application) + ' is waiting.')
        yield self.env.timeout(np.random.exponential(1 / self.queue_waiting_flow_rate))


def send_application(env, application, model):
    # Get current amount of applications in queue
    queue_applications_amount = len(model.channel.queue)
    # Get current amount of processing applications
    processing_applications_amount = model.channel.count

    model.total_qs_list.append(queue_applications_amount + processing_applications_amount)
    model.queue_list.append(queue_applications_amount)

    # Generate a request to use a channel
    with model.channel.request() as request:
        # print('Application ' + str(application) + ' is sent.')
        current_queue_len = len(model.channel.queue)
        # Number of users currently using the resource
        current_count_len = model.channel.count

        if current_queue_len <= model.max_queue_length:
            # Moment in time when application enters model
            start_time = env.now
            model.applications_done.append(current_queue_len + current_count_len)
            # An application ether waits for a channel to become free or starts to pro
            res = yield request | env.process(model.application_waiting(application))
            model.queue_times.append(env.now - start_time)
            if request in res:
                yield env.process(model.application_processing(application))
            model.total_wait_times.append(env.now - start_time)
            # print('Application ' + str(application) + ' is processed.')
        else:
            model.applications_rejected.append(channels_number + max_queue_length + 1)
            # print('Application ' + str(application) + ' is rejected.')
            model.queue_times.append(0)
            model.total_wait_times.append(0)


def run_model(env, model):
    application = 0

    while True:
        yield env.timeout(np.random.exponential(1 / model.applications_flow_rate))
        application += 1
        env.process(send_application(env, application, model))


def find_average_qs_len(total_qs_list):
    average_qs_len = np.array(total_qs_list).mean()
    print('Average amount of applications in QS is: ' + str(average_qs_len))
    return average_qs_len


def find_average_wait_time(total_wait_times):
    average_wait_time = np.array(total_wait_times).mean()
    print('Average time in QS is: ' + str(average_wait_time))
    return average_wait_time


def find_average_queue_len(queue_list):
    average_queue_len = np.array(queue_list).mean()
    print('Average queue length is: ' + str(average_queue_len))
    return average_queue_len


def find_average_queue_time(queue_times):
    average_queue_time = np.array(queue_times).mean()
    print('Average time in queue is: ' + str(average_queue_time))
    return average_queue_time


def find_empiric_probabilities(applications_done, applications_rejected, queue_times, total_wait_times, total_qs_list,
                               queue_list, num_channel,
                               max_queue_length,
                               applications_flow_rate,
                               service_flow_rate):
    print('-------------------Empiric---------------------')
    total_applications_amount = len(applications_done) + len(applications_rejected)
    P = []
    for value in range(1, num_channel + max_queue_length + 1):
        P.append(len(applications_done[applications_done == value]) / total_applications_amount)
    print('Empiric final probabilities:')
    for index, p in enumerate(P):
        print('P' + str(index) + ': ' + str(p))
    P_reject = len(applications_rejected) / total_applications_amount
    print('Empiric probability of rejection: ' + str(P_reject))
    Q = 1 - P_reject
    print('Empiric Q:' + str(Q))
    A = applications_flow_rate * Q
    print('Empiric A: ' + str(A))
    find_average_queue_len(queue_list)
    find_average_qs_len(total_qs_list)
    find_average_queue_time(queue_times)
    average_full_channels = Q * applications_flow_rate / service_flow_rate
    print('Empiric average amount of busy channels: ' + str(average_full_channels))
    find_average_wait_time(total_wait_times)
    axs[0].hist(total_wait_times, 50)
    axs[0].set_title('Wait times')
    axs[1].hist(total_qs_list, 50)


def find_theoretical_probabilities(num_channel, max_queue_length, applications_flow_rate, service_flow_rate,
                                   queue_waiting_flow_rate):
    print('-------------------Theoretical---------------------')
    ro = applications_flow_rate / service_flow_rate
    betta = queue_waiting_flow_rate / service_flow_rate
    print('Theoretical final probabilities:')
    P = 0
    p0 = (sum([ro ** i / factorial(i) for i in range(num_channel + 1)]) +
          (ro ** num_channel / factorial(num_channel)) *
          sum([ro ** i / (np.prod([num_channel + t * betta for t in range(1, i + 1)])) for i in
               range(1, max_queue_length + 1)])) ** -1
    print('P0: ' + str(p0))
    P += p0
    for i in range(1, num_channel + 1):
        px = (ro ** i / factorial(i)) * p0
        P += px
        print('P' + str(i) + ': ' + str(px))
    pn = px
    pq = px
    for i in range(1, max_queue_length):
        px = (ro ** i / np.prod([num_channel + t * betta for t in range(1, i + 1)])) * pn
        P += px
        if i < max_queue_length:
            pq += px
        print('P' + str(num_channel + i) + ': ' + str(px))
    P = px
    print('Theoretical probability of rejection: ' + str(P))
    Q = 1 - P
    print('Theoretical Q: ', Q)
    A = Q * applications_flow_rate
    print('Theoretical A: ', A)
    L_q = sum([i * pn * (ro ** i) / np.prod([num_channel + t * betta for t in range(1, i + 1)]) for
               i in range(1, max_queue_length + 1)])
    print('Average queue length is: ', L_q)
    L_pr = sum([index * p0 * (ro ** index) / factorial(index) for index in range(1, num_channel + 1)]) + sum(
        [(num_channel + index) * pn * ro ** index / np.prod(
            np.array([num_channel + t * betta for t in range(1, index + 1)])) for
         index in range(1, max_queue_length + 1)])
    print('Average amount of applications in QS is: ', L_pr)
    print('Average time in queue is: ', Q * ro / applications_flow_rate)
    print('Average amount of busy channels: ', Q * ro)
    print('Average time in QS is: ', L_pr / applications_flow_rate)


if __name__ == '__main__':
    print('First example:')
    channels_number = 2
    print('Amount of channels (n): ' + str(channels_number))
    service_flow_rate = 4
    print('Service flow rate (mu): ' + str(service_flow_rate))
    applications_flow_rate = 3
    print('Applications flow rate (lambda): ' + str(applications_flow_rate))
    queue_waiting_flow_rate = 1
    print('Queue waiting flow rate (v): ' + str(queue_waiting_flow_rate))
    max_queue_length = 2
    print('Max queue length (m): ' + str(max_queue_length))

    env = simpy.Environment()
    model = QueuingSystemModel(env, channels_number, service_flow_rate, applications_flow_rate, queue_waiting_flow_rate,
                               max_queue_length)
    print('Running simulation...')
    env.process(run_model(env, model))
    # How long the simulation will run
    env.run(until=100)

    find_empiric_probabilities(np.array(model.applications_done), np.array(model.applications_rejected),
                               np.array(model.queue_times),
                               np.array(model.total_wait_times),
                               np.array(model.total_qs_list),
                               np.array(model.queue_list), channels_number, max_queue_length, applications_flow_rate,
                               service_flow_rate)
    find_theoretical_probabilities(channels_number, max_queue_length, applications_flow_rate, service_flow_rate,
                                   queue_waiting_flow_rate)
    print('End of first example')
    plt.show()
