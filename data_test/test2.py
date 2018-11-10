from assignment2 import *
import datetime
data_folder = ""#"data_test\\"

'''
time=datetime.datetime.now()
list=['big.txt']
error_list=['data_test/commmon_errors.txt']
alex2=learn_language_model(list,n=3)
'''

def get_ngram(s, err, n):
    ngram = []
    index = s.index(err) - 1
    for i in range(n - 1):
        if index < 0:
            if index % 2 == 0:
                ngram.append("<END>")
            if index % 2 != 0:
                ngram.append("<START>")
        else:
            ngram.append(s[index])

        index -= 1
    ngram.reverse()
    ngram=tuple(ngram)
    return ngram

alex=get_ngram('alex is cool but i dont know what happended'.split(' '),'is',6)
print(alex)