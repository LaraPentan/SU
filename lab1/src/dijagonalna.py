from Klasifikator import *

import numpy

class dijagonalna(Klasifikator):
	def __init__(self):
		super(dijagonalna, self).__init__()
		return

	def train(self, train_set, poredak = None):
		super(dijagonalna, self).train(train_set, poredak)

		novi_sigma = None	

		for key in self.klase:
			if novi_sigma == None:
				novi_sigma = self.P[key] * self.sigma[key]
			else:
				novi_sigma += self.P[key] * self.sigma[key]
		
		(r, c) = novi_sigma.shape

		for i in xrange(r):
			for j in xrange(c):
				if i != j:
					novi_sigma.itemset((i, j), 0.0)

		for key in self.klase:
			self.sigma[key] = novi_sigma
	
		return				
