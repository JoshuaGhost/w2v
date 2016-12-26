class Combiner(object):

	def cbn_factory(self, combiner_type, num_sample_words, abs_sort, test_mode):
		if combiner_type == 0:
			return Cbn_sort(num_sample_words, abs_sort, test_mode)
