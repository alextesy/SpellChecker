from assignment2 import *
import datetime
data_folder = ""#"data_test\\"

'''
time=datetime.datetime.now()
list=['big.txt']
error_list=['data_test/commmon_errors.txt']
alex2=learn_language_model(list,n=3)
'''
list=['big.txt']


def evaluate_text(s, n, lm, ngram_dict):
    """Returns the likelihood of the specified sentence to be generated by the
    the specified language model. Using Laplace smoothing when needed.

    Args:
        s (str): the sentence to evaluate.
        n (int): the length of the n-grams to consider in the language model.
        lm (dict): the language model to evaluate the sentence by.

    Returns:
        The likelihood of the sentence according to the language model (float).

    Exceptions:
        If the sentence wasn't given the program shuts down

    """
    if not s or len(s) == 0:
        print('The sentence was not given')
        sys.exit()
    s = s.split(' ')
    s = [normalization(word) for word in s]
    prob = 0
    for i, word in enumerate(s):
        ngram = get_n_words(s, i, n)
        mehane = float(len(lm))
        if ngram in ngram_dict:
            mehane+=sum(ngram_dict[ngram].values())
        mone = 0.0
        if word in lm:
            if ngram in lm[word]:
                mone = float(lm[word][ngram])
        mone = mone + 1.0
        prob += math.log(mone / mehane)
    prob = math.exp(prob)
    return prob


lm=learn_language_model(list,n=3)


new_lm = defaultdict(lambda: defaultdict(int))
for k, v in lm.items():
    for words, count in v.items():
        new_lm[words][k] = count

new_ngram_dict = defaultdict(lambda: defaultdict(int))
for word, new_ngrams in new_lm.items():
    for ngram, count in new_ngrams.items():
        new_ngram_dict[ngram][word] = count



print(evaluate_text('when will your',3,new_lm,new_ngram_dict))
print(evaluate_text('when will you',3,new_lm,new_ngram_dict))

