def get_n_words(list,index,n):
    """

    :param list: A list of tokens
    :param index: current token index
    :param n: n of the ngram
    :return: the last n-1 words before the word in the given index
    """
    return_sent=[]
    for i in range(n-1,0,-1):
        if index-i<0:
            return_sent.append('')
        else:
            return_sent.append(list[index-i])
    return tuple(return_sent)

test=get_n_words(['alex','is','coo','and','is','war'],2,3)
print(test)