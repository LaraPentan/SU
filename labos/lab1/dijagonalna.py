from Klasifikator import *

import numpy

class dijagonalna(Klasifikator):
	def __init__(self):
		super(dijagonalna, self).__init__()
		self.mali_sigma = dict()

	def train(self, train_set, poredak = None):
		super(dijagonalna, self).train(train_set, poredak)
		
		for uzorak in train_set:
			(x, y) = uzorak
			for i in xrange(len(x)):
				if i in self.mali_sigma:
					self.mali_sigma[i] += numpy.power((x[i] - self.mi[y].item(i, 0)), 2.)
				else:
					self.mali_sigma[i] = numpy.power((x[i] - self.mi[y].item(i, 0)), 2.) 
			
		for key in self.mali_sigma:
			self.mali_sigma[key] /= 1.0 * len(train_set)

		return				

	def getProbability(self, x, klasa):
		P = 1.0
		for i in xrange(len(x)):
			P /= (numpy.power(2. * numpy.pi * self.mali_sigma[i], 0.5))
			P *= numpy.exp(-0.5 * numpy.power(x[i] - self.mi[klasa].item(i, 0), 2.) / self.mali_sigma[i]);
		return P
