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
		self.train_set = list()
	
	def train(self, train_set, poredak = None):
		self.train_set = train_set
		skup_klasa = set()

		# compute mi and P
		for uzorak in train_set:
			(x, y) = uzorak

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
		if poredak == None:
			self.klase = list(skup_klasa)
		else:
			self.klase = poredak
		
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

		# print computed values
		"""
		for key in self.klase:
			print "--------", key, "---------"
			print "P"
			print self.P[key]
			print "mi"
			print self.mi[key]
			print "sigma"
			print self.sigma[key]
			print "--------------------------"
		"""
		return

	def getProbability(self, x, klasa):
		# needs to be implemented in more specific class
		return -1

	def assignClass(self, x):
		# assume some random class is the best fit
		najbolja_klasa = set_head(self.klase)
		najbolja_ocjena = self.getProbability(x, najbolja_klasa)
 
		# find the best fit class
		for klasa in self.klase:
			ocjena = self.getProbability(x, klasa)
			if ocjena > najbolja_ocjena:
				najbolja_ocjena = ocjena
				najbolja_klasa = klasa

		return najbolja_klasa

	def getErrorRate(self, test_set):
		greska = 0.0
		uzoraka = len(test_set)

		if uzoraka == 0: return 0
		
		for uzorak in test_set:
			(x, y) = uzorak
			h = self.assignClass(x)
			if h != y:
				greska += 1.0

		return greska / uzoraka
	
	def printAposterioriValues(self, filename):
		f = open(filename, "w")
		# TODO: remove string formatting before submitting		

		buff = ""
		for klasa in self.klase:
			buff += ("%10s\t" % klasa)
		buff += "klasa" + '\n'
	
		f.write(buff)

		for uzorak in self.train_set:
			(x, y) = uzorak
			buff = ""
			for klasa in self.klase:
				buff += ("%10lf\t" % self.getProbability(x, klasa))
			buff += self.assignClass(x) + '\n'

			f.write(buff)

		f.close()
		return
