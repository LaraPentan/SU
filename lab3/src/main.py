# -*- coding: utf-8 -*-
import sys
import codecs
import numpy

import k_means
import em

def parse_input(filename):
    L = list()
    classes = set()

    f = open(filename, "r")
    feature_cnt = 0

    for line in f:
        xs = line.rstrip().split(" ")
        x = numpy.array(xs[:-1], dtype=float)

        feature_cnt = max(feature_cnt, len(x))

        y = xs[-1]

        L.append((x, y))
        classes.add(y)

    f.close()
    return (L, feature_cnt, classes)

def parse_configurations(filename):
    configurations = list()
    current_centroids = None

    f = open(filename, "r")

    for line in f:
        if line.startswith("Konfiguracija"):
            if current_centroids:
                configurations.append(current_centroids)

            current_centroids = list()
        elif line.rstrip():
            xs = line.rstrip().split(" ")
            x = numpy.array(xs[:-1], dtype=float)
            y = xs[-1]

            current_centroids.append((x, y))

    if len(current_centroids):
        configurations.append(current_centroids)

    f.close()

    return configurations

def print_k_means(input_data, feature_cnt, K, centroids, f, verbose = None):
    result = k_means.group(input_data, feature_cnt, K, centroids, verbose)
    if result == None:
        print "KMEANS returned invalid result!"
        return

    (groups, (iteracija, J)) = result

    f.write("K = %d\n" % (K))

    for (i, (group, cnt)) in zip(xrange(K), groups):
        f.write("c%d:" % (i + 1))
        for x in group:
            f.write(" %.2lf" % (x))

        f.write("\ngrupa %d: %d primjera\n" % (i + 1, cnt))

    f.write("#iter: %d\nJ: %.2lf\n" % (iteracija, J))
    return

def print_em(input_data, feature_cnt, K, centroids, f, verbose = None):
    result = em.group(input_data, feature_cnt, K, centroids, verbose)
    if result == None:
        print "EM returned invalid result!"
        return

    (groups, (iteracija, log_izglednost)) = result

    f.write("K = %d\n" % (K))

    for (i, (group, cnt)) in zip(xrange(K), groups):
        f.write("c%d:" % (i + 1))
        for x in group:
            f.write(" %.2lf" % (x))

        f.write("\ngrupa %d, %d primjera\n" % (i + 1, cnt))

    f.write("#iter: %d\nlog-izglednost: %.2lf\n" % (iteracija, log_izglednost))
    return

def main():
    # check if we're given enough cmd parameters
    if len(sys.argv) != 4:
        print "Zadan nedovoljan broj parametara"
        sys.exit(1)

    # prepare groups for k_means
    print "Parsing input data: ", sys.argv[1]
    (input_data, feature_cnt, classes) = parse_input(sys.argv[1])

    # prepare configurations for EM
    print "Parsing configurations: ", sys.argv[2]
    configurations = parse_configurations(sys.argv[2])

    # get output path
    OUTPUT_PATH = sys.argv[3]
    print "Output path: ", sys.argv[3]

    # KMEANS
    # open file
    kmeans_file = codecs.open(OUTPUT_PATH + "kmeans-all.dat", "w", "utf-8")

    # K = 2
    print "KMEANS: Computing groups for K = 2 ...",
    centroids = [input_data[15], input_data[4]]
    print_k_means(input_data, feature_cnt, len(centroids), centroids, kmeans_file)
    print "Done"

    kmeans_file.write("--\n")

    # K = 3
    print "KMEANS: Computing groups for K = 3 ...",
    centroids.append(input_data[0])
    print_k_means(input_data, feature_cnt, len(centroids), centroids, kmeans_file)
    print "Done"

    kmeans_file.write("--\n")

    # K = 4
    print "KMEANS: Computing groups for K = 4 ...",
    centroids.append(input_data[2])
    print_k_means(input_data, feature_cnt, len(centroids), centroids,
            kmeans_file, OUTPUT_PATH + "kmeans-k4.dat")
    print "Done"

    kmeans_file.write("--\n")

    # K = 5
    print "KMEANS: Computing groups for K = 5 ...",
    centroids.append(input_data[9])
    print_k_means(input_data, feature_cnt, len(centroids), centroids, kmeans_file)
    print "Done"

    # close file
    kmeans_file.close()

    # EM
    # open file
    em_file = codecs.open(OUTPUT_PATH + "em-all.dat", "w", "utf-8")

    # K = 2
    print "EM: Computing groups for K = 2 ...",
    centroids = [input_data[15], input_data[4]]
    print_em(input_data, feature_cnt, len(centroids), centroids, em_file)
    print "Done"

    em_file.write("--\n")

    # K = 3
    print "EM: Computing groups for K = 3 ...",
    centroids.append(input_data[0])
    print_em(input_data, feature_cnt, len(centroids), centroids, em_file)
    print "Done"

    em_file.write("--\n")

    # K = 4
    print "EM: Computing groups for K = 4 ...",
    centroids.append(input_data[2])
    print_em(input_data, feature_cnt, len(centroids), centroids,
            em_file, OUTPUT_PATH + "em-k4.dat")
    print "Done"

    em_file.write("--\n")

    # K = 5
    print "EM: Computing groups for K = 5 ...",
    centroids.append(input_data[9])
    print_em(input_data, feature_cnt, len(centroids), centroids, em_file)
    print "Done"

    # close file
    em_file.close()


    # EM with given different configurations
    # open file
    conf_file = codecs.open(OUTPUT_PATH + "em-konf.dat", "w", "utf-8")

    print "Running EM with different configurations"

    first = True
    for (i, config) in zip(xrange(len(configurations)), configurations):

        print "Running configuraiton ", i, "...",

        result = em.group(input_data, feature_cnt, len(config), config)
        if result == None:
            print "EM returned invalid result!"
            continue

        (groups, (iteracija, log_izglednost)) = result

        if first == False:
            conf_file.write("--\n")
        else:
            first = False

        conf_file.write("Konfiguracija %d:\n" % (i + 1))
        conf_file.write("log-izglednost: %.2lf\n" % (log_izglednost))
        conf_file.write("#iteracija: %d\n" % (iteracija))

        print "Done"

    # close file
    conf_file.close()

    # EM using k_means for initialization
    print "Using KMEANS to init EM...",

    # init with kmeans
    centroids = [input_data[0], input_data[2], input_data[4], input_data[15]]
    result = k_means.group(input_data, feature_cnt, 4, centroids)

    if result == None:
        print "KMEANS returned invalid result!"
        return

    (groups, (iteracija, J)) = result

    print "Done"
    print "Running EM...",

    # continue with em

    result = em.group(input_data, feature_cnt, 4, groups, verbose2 =
            OUTPUT_PATH + "em-kmeans.dat")

    if result == None:
        print "EM returned invalid result!"
    else:
        print "Done"

    # end
    return

if __name__=="__main__":
    numpy.seterr(all='ignore')
    main()
