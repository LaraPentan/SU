from Klasifikator import *

import numpy

class opceniti(Klasifikator):
	def __init__(self):
		super(opceniti, self).__init__()

	def getProbability(self, x, klasa):
		n = len(x)
		razlika = numpy.matrix(x).T - self.mi[klasa]

		determinanta = numpy.linalg.det(self.sigma[klasa])
		umnozak = razlika.T * self.sigma[klasa].I * razlika

		brojnik = numpy.exp(-0.5 * umnozak.item(0,0))
		nazivnik = numpy.sqrt(determinanta) * numpy.power((2 * numpy.pi), n / 2.)

		return brojnik / nazivnik

