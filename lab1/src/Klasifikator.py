import copy
import numpy

def set_head(s):
	return copy.copy(s).pop()

class Klasifikator(object):
	def __init__(self):
		self.P = dict()
		self.mi = dict()
		self.sigma = dict()
		self.klase = list()
		self.__cache = dict()

	def __dummy_train(self):
		# primjer iz skripte, 68.stranica
		self.klase = [1, 2]

		self.mi[1] = numpy.matrix([[1, -5, 0]]).T
		self.mi[2] = numpy.matrix([[7, 2, 3]]).T

		self.P[1] = 0.4
		self.P[2] = 0.6

		self.sigma[1] = numpy.matrix([[1, -0.75, 0.2], [-0.75, 6.25, 1.5], [0.2, 1.5, 4]])
		self.sigma[2] = numpy.matrix([[4, 0.7, 2], [0.7, 12.25, -3.5], [2, -3.5, 16]])
		return

	def train(self, train_set, poredak = list()):
		skup_klasa = set()

		# compute mi and P
		for uzorak in train_set:
			(x, y) = uzorak

			if y not in poredak:
				skup_klasa.add(y)

			if y in self.P:
				self.P[y] += 1.0
			else:
				self.P[y] = 1.0

			if y in self.mi:
				self.mi[y] += numpy.matrix(x).T
			else:
				self.mi[y] = numpy.matrix(x).T

		# extract classes into list
		self.klase = poredak + sorted(list(skup_klasa))

		# fix computed values
		for key in self.klase:
			self.mi[key] /= self.P[key]

		# compute sigma
		for uzorak in train_set:
			(x, y) = uzorak

			razlika = numpy.matrix(x).T - self.mi[y]

			if y in self.sigma:
				self.sigma[y] += razlika * razlika.T
			else:
				self.sigma[y] = razlika * razlika.T


		# fix computed values
		for key in self.klase:
			self.sigma[key] /= self.P[key]

		# fix computed values
		for key in self.klase:
			self.P[key] /= 1.0 * len(train_set)

		return

	def getProbability(self, x, klasa):
		n = len(x)
		razlika = numpy.matrix(x).T - self.mi[klasa]

		determinanta = numpy.linalg.det(self.sigma[klasa])
		umnozak = razlika.T * self.sigma[klasa].I * razlika

		brojnik = numpy.exp(-0.5 * umnozak.item(0, 0))
		nazivnik = numpy.sqrt(determinanta) * numpy.power(2. * numpy.pi, n / 2.)

		return brojnik / nazivnik

	def assignClass(self, x):
		najbolja_klasa = set_head(self.klase)
		najbolja_ocjena = self.getAposterioriValue(najbolja_klasa, x)

		for klasa in self.klase:
			ocjena = self.getAposterioriValue(klasa, x)
			if ocjena > najbolja_ocjena:
				najbolja_ocjena = ocjena
				najbolja_klasa = klasa

		return (najbolja_klasa, najbolja_ocjena)

	def getErrorRate(self, test_set):
		greska = 0.0
		uzoraka = len(test_set)

		if uzoraka == 0: return 0

		for uzorak in test_set:
			(x, y) = uzorak
			(h, p) = self.assignClass(x)
			if h != y:
				greska += 1.0

		return greska / uzoraka

	def getAposterioriValue(self, klasa, x):
		if (klasa, tuple(x)) in self.__cache:
			return self.__cache[(klasa, tuple(x))]

		Px = 0.0
		for key in self.klase:
			Px += self.P[key] * self.getProbability(x, key)

		prob = self.P[klasa] * self.getProbability(x, klasa) / Px
		self.__cache[(klasa, tuple(x))] = prob
		return prob

	def getAmbigousSamples(self, test_set):
		sve = list()
		ret = list()

		for uzorak in test_set:
			(x, y) = uzorak
			(h, p) = self.assignClass(x)

			sve.append((p, uzorak))

		for r in sorted(sve):
			(ocjena, uzorak) = r
			ret.append(uzorak)

		return ret

	def printParameters(self):
		print ""
		for key in self.klase:
			print "--------", key, "---------"
			print "P"
			print self.P[key]
			print "mi"
			print self.mi[key]
			print "sigma"
			print self.sigma[key]
			print "--------------------------"
		return

	def printAposterioriValues(self, test_set, filename):
		f = open(filename, "w")

		buff = ""
		for klasa in self.klase:
			buff += ("%s\t" % klasa)
		buff += "klasa" + '\n'

		f.write(buff)

		for uzorak in test_set:
			(x, y) = uzorak
			(h, p) = self.assignClass(x)

			buff = ""
			for klasa in self.klase:
				buff += ("%.2lf\t" % self.getAposterioriValue(klasa, x))

			buff += h + '\n'

			f.write(buff)

		f.close()
		return
