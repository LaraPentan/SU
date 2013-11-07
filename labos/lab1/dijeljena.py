from Klasifikator import *

import numpy

class dijeljena(Klasifikator):
	def __init__(self):
		super(dijeljena, self).__init__()
		self.novi_sigma = None

	def train(self, train_set, poredak = None):
		super(dijeljena, self).train(train_set, poredak)
	
		for key in self.klase:
			if self.novi_sigma == None:
				self.novi_sigma = self.P[key] * self.sigma[key]
			else:
				self.novi_sigma += self.P[key] * self.sigma[key]

	def getProbability(self, x, klasa):
		n = len(x)
		razlika = numpy.matrix(x).T - self.mi[klasa]

		determinanta = numpy.linalg.det(self.novi_sigma)
		umnozak = razlika.T * self.novi_sigma.I * razlika

		brojnik = numpy.exp(-0.5 * umnozak.item((0, 0)))
		nazivnik = (numpy.sqrt(determinanta) * numpy.power((2 * numpy.pi), n / 2.)) 

		return brojnik / nazivnik

		
