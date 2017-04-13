from textblob import TextBlob
import string
import gensim
import numpy
import xarray
import numba

class Corpus:
	def __init__(self, path):

		self.content = TextBlob(open(path).read().translate(str.maketrans("", "", ",-'"))).lower()
		self.counts = self.content.word_counts
		self.mapping = gensim.corpora.dictionary.Dictionary([self.content.tokens]).token2id
		self.sentences = self.content.sentences
		self.language_model = {k: v/sum(self.counts.values()) for k, v in self.counts.items()}

	def _numerize(self):

		all_sentences = numpy.array([xarray.DataArray([self.mapping[i] for i in self.sentences[m].tokens[0:-1]],
			            coords={'Tokens': self.sentences[m].tokens[0:-1]}, dims=('Tokens'))
			            for m in range(0, len(self.sentences))])
		return all_sentences

	def structure(self, other):

		@numba.jit()
		def calculate(A, B):
			pass

		for i, k in zip(self._numerize(), other._numerize()):
			pass

		return xarray.DataArray(numpy.zeros((len(self.mapping)-1, len(other.mapping)-1)), dims=('English', 'French'))


Eng = Corpus('C:/Users/Vladimir/Desktop/NLP/test.txt')
Fr = Corpus('C:/Users/Vladimir/Desktop/NLP/Frtest.txt')

print(Eng.structure(Fr))




