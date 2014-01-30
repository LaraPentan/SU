# -*- coding: utf-8 -*-
import math
import numpy
import random
import codecs

import sys

from itertools import groupby

EPS = 1e-5

def get_probability(x, mi, sigma):
    n = len(x)
    razlika = numpy.matrix(x - mi).T

    determinanta = numpy.linalg.det(sigma)
    umnozak = (razlika.T * sigma.I * razlika).item(0, 0)

    brojnik = numpy.exp(-0.5 * umnozak)
    nazivnik = numpy.sqrt(determinanta) * numpy.power(2. * numpy.pi, n / 2.)

    return brojnik / nazivnik

def get_log_likelihood(D, phi):
    res = 0.0

    for (x, y) in D:
        acc = 0.0

        for (pi_k, mi_k, sigma_k) in phi:
            acc += pi_k * get_probability(x, mi_k, sigma_k)

        res += math.log(acc)

    return res

def group(data, feature_cnt, K = 4, centroids = None, verbose = None, verbose2 = None):
    # verbose output file
    fajl = None
    if verbose:
        fajl = codecs.open(verbose, "w", "utf-8")

    fajl2 = None
    if verbose2:
        fajl2 = codecs.open(verbose2, "w", "utf-8")
        fajl2.write("#iteracije: log-izglednost\n--\n")

    # global constants
    global EPS

    N = len(data)
    result = None

    # if we didn't specify centroids, pick random K from data
    if not centroids:
        centroids = random.sample(data, K)

    # separate class name from centroids
    classes   = [y for (x, y) in centroids]
    centroids = [x for (x, y) in centroids]

    # podaci o algoritmu
    iteracija = 1

    # inicijalizacija parametara
    phi = list()
    for i in xrange(K):
        pi_k = 1. / K
        mi_k = numpy.matrix(centroids[i])
        sigma_k = numpy.matrix(numpy.identity(feature_cnt))
        phi.append((pi_k, mi_k, sigma_k))

    log_izglednost = get_log_likelihood(data, phi)

    # lets group this data
    while True:

        if math.isnan(log_izglednost):
            break

        if verbose2:
            fajl2.write("#%d: %.2lf\n" % (iteracija - 1, log_izglednost))

        num_centroids = dict()

        # E korak
        H = dict()
        for k in xrange(K):
            H[k] = list()
            num_centroids[k] = list()

        for (x, y) in data:
            max_h = -1
            max_id = None

            suma = 0.0
            prob = list()

            for (pi_k, mi_k, sigma_k) in phi:
                val = get_probability(x, mi_k, sigma_k) * pi_k

                suma += val
                prob.append(val)

            for (k, p) in zip(xrange(K), prob):
                val = p / suma

                if max_id == None or val > max_h:
                    max_h = val
                    max_id = k

                H[k].append(val)

            num_centroids[max_id].append((max_h, y))

        # M korak
        new_phi = list()

        for k in xrange(K):
            h = H[k]
            sum_h = sum(h)

            # init new values
            new_pi = 0.0
            new_mi = numpy.matrix(numpy.zeros(feature_cnt, dtype=float))
            new_sigma = numpy.matrix(numpy.zeros((feature_cnt, feature_cnt)))

            # compute new values
            for (h_i, (x, y)) in zip(h, data):
                new_mi += h_i * x

            new_mi /= sum_h

            for (h_i, (x, y)) in zip(h, data):
                razlika = numpy.matrix(x - new_mi).T
                new_sigma += h_i * (razlika * razlika.T)

            new_sigma /= sum_h

            new_pi = sum_h / N

            # append new values
            new_phi.append((new_pi, new_mi, new_sigma))

        # compute new log likelihood
        new_log_izglednost = get_log_likelihood(data, new_phi)

        if math.fabs(new_log_izglednost - log_izglednost) < EPS:
            # return the result
            result = [numpy.squeeze(numpy.asarray(b)) for (a, b, c) in new_phi]

            if verbose2:
                fajl2.write("--\n")

            counter = list()
            first = True
            for k in xrange(K):

                if verbose:
                    if first == False:
                        fajl.write("--\n")
                    else:
                        first = False

                    fajl.write("Grupa %d:\n" % (k + 1))

                lista = list()

                if k in num_centroids:
                    counter.append(len(num_centroids[k]))
                    lista = num_centroids[k]
                else:
                    counter.append(0)

                if verbose:
                    lista = sorted(lista, key=lambda (a, b): -a)
                    for (val, name) in lista:
                        fajl.write("%s %.2lf\n" % (name, val))
                # endif

                if verbose2:
                    fajl2.write("Grupa %d: " % (k + 1))

                    lista = [name for (val, name) in lista]
                    lista = sorted([(len(list(group)), key) for key, group in
                        groupby(sorted(lista))], reverse=True)

                    first = True
                    for (cnt, name) in lista:
                        if not first:
                            fajl2.write(",")
                        else:
                            first = False

                        fajl2.write(" %s %d" % (name, cnt))

                    fajl2.write("\n")
                #endif

            result = (zip(result, counter), (iteracija, new_log_izglednost))
            break
        else:
            log_izglednost = new_log_izglednost
            phi = new_phi

        iteracija += 1

    # write result to file if needed
    if verbose:
        fajl.close()

    if verbose2:
        fajl2.close()

    return result

