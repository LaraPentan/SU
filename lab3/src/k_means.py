# -*- coding: utf-8 -*-
import math
import numpy
import random
import codecs

EPS = 1e-9

def group(data, feature_cnt, K = 4, centroids = None, verbose = None):
    # verbose output file
    fajl = None
    if verbose:
        fajl = codecs.open(verbose, "w", "utf-8")
        fajl.write("#iteracije: J\n--\n")

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

    # lets group this data
    while iteracija < 300:
        B = list()

        for (x, y) in data:
            indicate = numpy.zeros(K, dtype=bool)

            best_id = -1
            best_val = 0

            for (i, mi) in zip(xrange(K), centroids):
                norma = numpy.dot(x - mi, x - mi)
                if best_id == -1 or (best_val > norma):
                    best_val = norma
                    best_id = i

            indicate[best_id] = True
            B.append(indicate)

        new_centroids = list()
        num_centroids = list()

        for i in xrange(K):
            suma = numpy.zeros(feature_cnt, dtype=float)
            numb = 0.0

            for (b, (x, y)) in zip(B, data):
                if b.item(i) == True:
                    suma += x
                    numb += 1.0

            if numb > 0:
                new_centroids.append(suma / numb)
                num_centroids.append(numb)
            else:
                new_centroids.append(0)
                num_centroids.append(0)

        # calculate grouping error
        J = 0.0
        for (b, (x, y)) in zip(B, data):
            for (i, mi) in zip(xrange(K), centroids):
                if b.item(i) == True:
                    J += numpy.dot(x - mi, x - mi)

        if verbose:
            fajl.write("#%d: %.2lf\n" % (iteracija - 1, J))

        # check convergence
        convergence = True
        for (old, new) in zip(centroids, new_centroids):
            for i in xrange(feature_cnt):
                if math.fabs(old[i] - new[i]) > EPS:
                    convergence = False
                    break

        if convergence:
            # return the result
            result = (zip(centroids, num_centroids), (iteracija, J))
            break
        else:
            centroids = new_centroids

        iteracija += 1

    # write result to file if needed
    if verbose:
        fajl.write("--\n")

        for i in xrange(K):
            fajl.write("Grupa %d:" % (i + 1))

            class_counter = dict()
            for (b, (x, y)) in zip(B, data):
                if b.item(i) == True:
                    if y in class_counter:
                        class_counter[y] += 1
                    else:
                        class_counter[y] = 1

            lista = sorted([(y, x) for x, y in class_counter.iteritems()], reverse=True)
            first = True
            for (cnt, name) in lista:
                if not first:
                    fajl.write(",")
                else:
                    first = False

                fajl.write(" %s %d" % (name, cnt))

            fajl.write("\n")
        fajl.close()

    return result
