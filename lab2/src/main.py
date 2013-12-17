# -*- coding: utf-8 -*-
import sys
import codecs
import numpy

from logisticka_regresija import *

def parse_input(filename, NUM):
    L = list()
    f = open(filename, "r")

    for line in f:
        x = list()

        xs = line.rstrip().split(" ")
        y = int(xs.pop(0))

        x = numpy.zeros(NUM)

        for uzorak in xs:
            (key, value) = uzorak.split(":")
            x[int(key)] = float(value)

        L.append((x, y))

    f.close()
    return L

def parse_dict(filename):
    D = dict()
    f = codecs.open(filename, "r", "utf-8")

    line_cnt = 0
    for line in f:
        (word, cnt) = line.rstrip().split(" ")
        D[line_cnt] = word
        line_cnt += 1

    f.close()
    return D

def main():
    # check if we're given enough cmd parameters
    if len(sys.argv) != 6:
        print "Zadan nedovoljan broj parametara"
        sys.exit(1)

    # read the dictionary
    print "Rjecnik: ", sys.argv[1]
    rjecnik = parse_dict(sys.argv[1])

    # number of features
    NUM = len(rjecnik)

    # read training set
    training_set = parse_input(sys.argv[2], NUM)

    # read test set
    test_set = parse_input(sys.argv[3], NUM)

    # read check set
    check_set = parse_input(sys.argv[4], NUM)

    # get output path
    OUTPUT_PATH = sys.argv[5]
    print "Output path: ", sys.argv[5]

    # <zadatak a>
    print "Zadatak a)"

    print "Racunam tezine razdvojnog vektora... ",
    (w0, w) = compute_weight_vector(training_set, 0, NUM)
    cee = CEE(training_set, w, w0, 0)
    ee = error_rate(training_set, w, w0)

    print "Done, cee = ", cee

    print "Ispisujem tezine u tezine1.dat...",
    f = codecs.open(OUTPUT_PATH + "tezine1.dat", "w", "utf-8")
    f.write("%.2lf\n" % w0)
    for wi in w:
        f.write("%.2lf\n" % wi)
    f.write("EE: %.2lf\n" % ee);
    f.write("CEE: %.2lf\n" % cee);

    f.close()
    print "Done, cee =", cee, ", ee =", ee
    # </zadatak a>

    # zadatak b
    print "Zadatak b)"
    faktori = [0, 0.1, 1, 5, 10, 100, 1000]

    optimalni_faktor = 0
    optimalni_ge = -1

    f = codecs.open(OUTPUT_PATH + "optimizacija.dat", "w", "utf-8")
    for faktor in faktori:
        print "Racunam tezine razdvojnog vektora za lambda = ", faktor, "...",
        (w0, w) = compute_weight_vector(training_set, faktor, NUM)

        trenutni_ge = error_rate(test_set, w, w0)
        trenutni_cee = CEE(test_set, w, w0, faktor)

        print "Done, cee = ", trenutni_cee, "ge = ", trenutni_ge

        f.write(u"\u03BB = %s, %.2lf\n" % (faktor, trenutni_ge))

        if optimalni_ge == -1 or optimalni_ge >= trenutni_ge:
            optimalni_ge = trenutni_ge
            optimalni_faktor = faktor

    f.write(u"optimalno: \u03BB = %s\n" % optimalni_faktor);
    f.close()

    print "Optimalni regularizacijski faktor = ", optimalni_faktor

    # </zadatak b>

    # <zadatak c>
    print "Zadatak c)"
    print "Racunam tezine razdvojnog vektora za lambda = ", optimalni_faktor, "...",
    (w0, w) = compute_weight_vector(training_set + test_set, optimalni_faktor, NUM)
    cee = CEE(training_set + test_set, w, w0, optimalni_faktor)
    ee = error_rate(training_set + test_set, w, w0)

    print "Done, CEE = ", cee, "ee =", ee

    print "Ispisujem tezine u tezine2.dat...",
    f = codecs.open(OUTPUT_PATH + "tezine2.dat", "w", "utf-8")

    f.write("%.2lf\n" % w0)
    for wi in w:
        f.write("%.2lf\n" % wi)
    f.write("EE: %.2lf\n" % ee);
    f.write("CEE: %.2lf\n" % cee);

    f.close()
    print "Done"

    print "Ispisujem 20 najznacajnjih rijeci u rijeci.txt...",
    f = codecs.open(OUTPUT_PATH + "rijeci.txt", "w", "utf-8")
    for indeks in map(lambda (x,y): y, sorted(zip(w, xrange(NUM)))[::-1])[:20]:
        f.write("%s\n" % rjecnik[indeks])

    f.close()
    print "Done"

    print "Ispisujem klasifikaciju ispitnog skupa u ispitni_predikcije.dat...",
    f = codecs.open(OUTPUT_PATH + "ispitni_predikcije.dat", "w", "utf-8")
    for sample in check_set:
        (x, y) = sample
        h = sigmoid( numpy.dot(w, x) + w0 )

        if h > 0.5: f.write("1\n")
        else: f.write("0\n")

    f.write(u"Greška: %.2lf\n" % error_rate(check_set, w, w0))
    f.close()
    print "Done"

    # <zadatak c/>

    return

if __name__ == "__main__":
	main()
