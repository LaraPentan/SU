import numpy
import math

def sigmoid(x):
    # avoiding
    if x > 20: return 1 - 1e-9
    if x < -20: return 1e-9
    return 1 / (1 + math.exp(-x))

def CEE(data_set, omega, omega_nula):
    E = 0.0
    for sample in data_set:
        (x, y) = sample
        h = sigmoid( numpy.dot(omega, x) + omega_nula )
        E += y * math.log(h) * (1 - y) * math.log(1 - h)

    return -E

def error_rate(data_set, omega, omega_nula):
    E = 0.0

    for sample in data_set:
        (x, y) = sample
        h = sigmoid( numpy.dot(omega, x) + omega_nula )

        if h > 0.5 and y == 0: E += 1
        if h < 0.5 and y == 1: E += 1

    return E / len(data_set)

def compute_weight_vector(data_set, regularizacija, feature_cnt):
    omega = numpy.zeros(feature_cnt)
    omega_nula = 0

    curr_cee = last_cee = -1

    while last_cee == -1 or math.fabs(last_cee - curr_cee) > 0.001:
        # init gradient descent
        delta_omega_nula = 0
        delta_omega = numpy.zeros(feature_cnt)

        for sample in data_set:
            (x, y) = sample
            h = sigmoid( numpy.dot(omega, x) + omega_nula)

            delta_omega_nula += h - y
            delta_omega += (h - y) * x

        # line search

        last = CEE(data_set, omega, omega_nula)
        eta = 0.01
        while eta <= 1.0:
            curr = CEE(data_set, omega * (1 - eta * regularizacija) - eta * delta_omega, omega_nula - eta * delta_omega_nula)

            if curr > last: break

            last = curr
            eta += 0.01

        # save solution
        curr_cee = last

        # perform gradient descent
        omega_nula -= eta * delta_omega_nula
        omega = omega * (1 - eta * regularizacija) - eta * delta_omega

        # update cee value
        last_cee = curr_cee

    return (omega_nula, omega)
