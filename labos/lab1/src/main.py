# -*- coding: utf-8 -*-
import sys
import codecs

from opceniti import *
from dijeljena import *
from dijagonalna import *
from izotropna import *

def parse_input(filename):
	L = list()
	f = open(filename, "r")

	# ignore first line
	f.readline()

	for line in f:
		x = list()

		xs = line.rstrip().split(" ")
		y = xs.pop()

		for uzorak in xs:
			x.append(float(uzorak))

		L.append((x, y))

	f.close()
	return L

def main():
	train_set = list()
	test_set = list()

	for arg in sys.argv:
		if arg[:7] == "-input0":
			print "Ucitavam skup za ucenje ...",
			train_set = parse_input(arg[8:])
			print "Done"
		elif arg[:7] == "-input1":
			print "Ucitavam skup za testiranje ...",
			test_set = parse_input(arg[8:])
			print "Done"

	# zeljeni poredak u ispisu
	poredak = ["Narančasta", "Žuta", "Zelena", "Plava", "Tirkizna", "Indigo", "Modra", "Magenta"]

	f = open("../output/greske.dat", "w")

	# opceniti
	model_a = opceniti()
	print "Treniram opceniti model ...",
	model_a.train(train_set, poredak)
	# model_a.printParameters()
	print "Done"
	print "Generiram izlaz za opceniti model ...",
	model_a.printAposterioriValues(test_set, "../output/opceniti.dat")
	f.write("%s\t%.2lf\t%.2lf\n" % ("opceniti", model_a.getErrorRate(train_set), model_a.getErrorRate(test_set)))
	print "Done"

	# dijeljena
	model_b = dijeljena()
	print "Treniram model s dijeljenom matricom ...",
	model_b.train(train_set, poredak)
	# model_b.printParameters()
	print "Done"
	print "Generiram izlaz za model s dijeljenom matricom ...",
	model_b.printAposterioriValues(test_set, "../output/dijeljena.dat")
	f.write("%s\t%.2lf\t%.2lf\n" % ("dijeljena", model_b.getErrorRate(train_set), model_b.getErrorRate(test_set)))
	print "Done"

	# dijagonalna
	model_c = dijagonalna()
	print "Treniram model s dijagonalnom matricom ...",
	model_c.train(train_set, poredak)
	# model_c.printParameters()
	print "Done"
	print "Generiram izlaz za model s dijagonalnom matricom ...",
	model_c.printAposterioriValues(test_set, "../output/dijagonalna.dat")
	f.write("%s\t%.2lf\t%.2lf\n" % ("dijagonalna", model_c.getErrorRate(train_set), model_c.getErrorRate(test_set)))
	print "Done"

	# izotropna
	model_d = izotropna()
	print "Treniram model s izotropnom matricom ...",
	model_d.train(train_set, poredak)
	# model_d.printParameters()
	print "Done"
	print "Generiram izlaz za model s izotropnom matricom ...",
	model_d.printAposterioriValues(test_set, "../output/izotropna.dat")
	f.write("%s\t%.2lf\t%.2lf\n" % ("izotropna", model_d.getErrorRate(train_set), model_d.getErrorRate(test_set)))
	print "Done"

	f.close()

	# nadji 5 najnejednoznacnijih tocaka za opceniti model
	print "Trazim nejednoznacne uzorke iz skupa za testiranje ...",
	nejednoznacne = model_a.getAmbigousSamples(test_set)
	print "Done"
	model_a.printAposterioriValues(nejednoznacne[:5], "../output/nejednoznacne.dat")

	return

if __name__ == "__main__":
	main()
