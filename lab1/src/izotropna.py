from Klasifikator import *

import numpy

class izotropna(Klasifikator):
	def __init__(self):
		super(izotropna, self).__init__()
		return

	def train(self, train_set, poredak = None):
		super(izotropna, self).train(train_set, poredak)
		novi_sigma = None	

		for key in self.klase:
			if novi_sigma == None:
				novi_sigma = self.P[key] * self.sigma[key]
			else:
				novi_sigma += self.P[key] * self.sigma[key]
		
		(r,c) = novi_sigma.shape
		procjena = 0.0

		for i in xrange(r):
			procjena += novi_sigma.item(i, i)

		procjena /= 1.0 * r

		for i in xrange(r):
			for j in xrange(c):
				if i == j:
					novi_sigma.itemset((i, j), procjena)
				else:
					novi_sigma.itemset((i, j), 0.0)

		for key in self.sigma:
			self.sigma[key] = novi_sigma

		return				
