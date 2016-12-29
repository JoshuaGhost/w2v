from Cbn_sort import Cbn_sort
class Combiner(object):
	@classmethod
	def cbn_factory(self, combiner_type, num_sample_words, abs_sort, test_mode):
		if combiner_type == 0:
			return Cbn_sort(num_sample_words, abs_sort, test_mode)
