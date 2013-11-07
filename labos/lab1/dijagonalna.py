from Klasifikator import *

import numpy

class dijagonalna(Klasifikator):
	def __init__(self):
		super(dijagonalna, self).__init__()
		self.mali_sigma = dict()

	def train(self, train_set, poredak = None):
		super(dijagonalna, self).train(train_set, poredak)
		
		for uzorak in train_set:
			(xs, y) = uzorak
			for i in xrange(len(xs)):
				if i in self.mali_sigma:
					self.mali_sigma[i] += numpy.power((xs[i] - self.mi[y].item(i, 0)), 2)
				else:
					self.mali_sigma[i] = numpy.power((xs[i] - self.mi[y].item(i, 0)), 2) 
			
		for key in self.mali_sigma:
			self.mali_sigma[key] /= len(train_set)

		return				

	def getProbability(self, x, klasa):
		P = 1.0
		for i in xrange(len(x)):
			P /= (self.mali_sigma[i] * numpy.power(2 * numpy.pi, 0.5))
			P *= numpy.exp(-0.5 * numpy.power(((x[i] - self.mi[klasa].item(i, 0)) / self.mali_sigma[i]), 2));
		return P
