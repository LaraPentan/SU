import numpy
import math

def sigmoid(x):
    # avoiding math errors
    if x > 25: return 1 - 1e-12
    if x < -25: return 1e-12
    return 1 / (1 + math.exp(-x))

def CEE(data_set, omega, omega_nula, regularizacija = 0):
    E = 0.0
    for sample in data_set:
        (x, y) = sample
        h = sigmoid( numpy.dot(omega, x) + omega_nula )

        E += y * math.log(h) + (1 - y) * math.log(1 - h)

    REG = numpy.dot(omega, omega) * regularizacija / 2.0

    return REG - E

def error_rate(data_set, omega, omega_nula):
    E = 0.0
    for sample in data_set:
        (x, y) = sample
        h = sigmoid( numpy.dot(omega, x) + omega_nula )

        if h > 0.5:
            if y == 0: E += 1.0
        else:
            if y == 1: E += 1.0

    return E / len(data_set)

def compute_weight_vector(data_set, regularizacija, feature_cnt):
    omega = numpy.zeros(feature_cnt)
    omega_nula = 0

    curr_cee = last_cee = -1

    while last_cee == -1 or math.fabs(last_cee - curr_cee) > 0.001:
        # update cee value
        last_cee = curr_cee

        # init gradient descent
        delta_omega = numpy.zeros(feature_cnt)
        delta_omega_nula = 0

        for sample in data_set:
            (x, y) = sample
            h = sigmoid( numpy.dot(omega, x) + omega_nula)

            delta_omega_nula += h - y
            delta_omega += (h - y) * x

        # line search
        eta = 0.0
        last = CEE(data_set, omega, omega_nula, regularizacija)

        while eta <= 1.0:
            novi_eta = eta + 0.01
            curr = CEE( data_set,
                        omega * (1.0 - novi_eta * regularizacija) - novi_eta * delta_omega,
                        omega_nula - novi_eta * delta_omega_nula,
                        regularizacija)

            if curr > last: break
            last = curr
            eta = novi_eta

        # update cee
        curr_cee = last

        # perform gradient descent
        omega_nula = omega_nula - eta * delta_omega_nula
        omega = omega * (1 - eta * regularizacija) - eta * delta_omega

    return (omega_nula, omega)
