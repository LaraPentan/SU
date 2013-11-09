from Klasifikator import *

import numpy

class dijeljena(Klasifikator):
	def __init__(self):
		super(dijeljena, self).__init__()
		return

	def train(self, train_set, poredak = None):
		super(dijeljena, self).train(train_set, poredak)
		novi_sigma = None	

		for key in self.klase:
			if novi_sigma == None:
				novi_sigma = self.P[key] * self.sigma[key]
			else:
				novi_sigma += self.P[key] * self.sigma[key]
		
		for key in self.sigma:
			self.sigma[key] = novi_sigma

		return
