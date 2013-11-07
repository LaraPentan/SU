from Klasifikator import *

import numpy

class izotropna(Klasifikator):
	def __init__(self):
		super(izotropna, self).__init__()
		self.mali_sigma = 0.0

	def train(self, train_set, poredak = None):
		super(izotropna, self).train(train_set, poredak)
		
		for uzorak in train_set:
			(xs, y) = uzorak
			novi_sigma = 0.0
			for i in xrange(len(xs)):
				novi_sigma += numpy.power((xs[i] - self.mi[y].item(i, 0)), 2)

			self.mali_sigma += novi_sigma / len(xs)

		self.mali_sigma /= len(train_set)
			
		return				

	def getProbability(self, x, klasa):
		P = 1.0
		for i in xrange(len(x)):
			P /= (self.mali_sigma * numpy.power(2 * numpy.pi, 0.5))
			P *= numpy.exp(-0.5 * numpy.power(((x[i] - self.mi[klasa].item(i, 0)) / self.mali_sigma), 2));
		return P
