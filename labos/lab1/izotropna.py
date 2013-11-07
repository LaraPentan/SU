from Klasifikator import *

import numpy

class izotropna(Klasifikator):
	def __init__(self):
		super(izotropna, self).__init__()
		self.novi_sigma = 0.0

	def train(self, train_set, poredak = None):
		super(izotropna, self).train(train_set, poredak)
		
		mali_sigma = dict()

		for uzorak in train_set:
			(x, y) = uzorak
			for i in xrange(len(x)):
				if i in mali_sigma:
					mali_sigma[i] += numpy.power((x[i] - self.mi[y].item(i, 0)), 2.)
				else:
					mali_sigma[i] = numpy.power((x[i] - self.mi[y].item(i, 0)), 2.)

		for key in mali_sigma:
			mali_sigma[key] /= 1.0 * len(train_set)

		for key in mali_sigma:
			self.novi_sigma += mali_sigma[key]

		self.novi_sigma /= 1.0 * len(mali_sigma)
			
		return				

	def getProbability(self, x, klasa):
		P = 1.0
		for i in xrange(len(x)):
			P /= (numpy.power(2. * numpy.pi * self.novi_sigma, 0.5))
			P *= numpy.exp(-0.5 * numpy.power(x[i] - self.mi[klasa].item(i, 0), 2.) / self.novi_sigma);
		return P
