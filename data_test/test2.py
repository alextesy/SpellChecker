from assignment2 import *


def learn_language_model(files, n=3, lm=None):
    words = '<END> <START> w1 w2 w3 w1 w2 w4 <END> <START>'

    words = words.split(' ')

    ngrams_list = ngrams(words, n)
    ngram_dict = defaultdict(lambda: defaultdict(int))
    for grams in ngrams_list:
        ngram_dict[tuple(grams[:-1])][grams[-1]] += 1
    ngram_dict = default_to_regular(ngram_dict)

alex=create_error_distribution()
print(alex)