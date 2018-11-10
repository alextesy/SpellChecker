import codecs
import collections
import math
import operator
import re
import sys
from collections import defaultdict,Counter

from nltk import ngrams,tokenize
from nltk.ccg.chart import lex
from nltk.misc import word_finder


def proccess_file(file,flag=None):
    '''
    :param file: full path of a file
    :return: a string of the content of the file
    '''
    try:
        with codecs.open(file, encoding='utf-8') as myfile:
            if not flag:
                return myfile.read().replace('\n', ' ')
            else:
                return myfile.readlines()

    except IOError:
        print("Could not read file:", file)
        sys.exit()

def normalization(word):
    """
    The normalization rules:
    1. Numbers 1-10 to strings - I found that we have multiple (hundreds and with some numbers thousands)
     cases of different representation of the number 1 and one, 2 and two, so i combined them.
    2. I throw away all the strings that contain numbers and letters - most of them were gibrish
    3. I turn every token to low case
    4. I throw away most of the 1 length words.
    5. I throw away some words that have hundreds of mentions but doesnt add anyting to LM (like 'www','http' etc)


    :param word:a single token
    :return: a token after normalization or None if not relevant
    """

    '''Normalizing tokens'''
    #Normalizing numbers
    if not word:
        return word
    word= str(word)
    if RepresentsInt(word):
        switch={0:'zero,',
                1:'one',
                2:'two',
                3:'three',
                4:'four',
                5:'five',
                6:'six',
                7:'seven',
                8:'eight',
                9:'nine',
                10:'ten'}
        if word in switch:
            return switch[word]
        return word
    if not check_numbers_letters(word):
        return None
    word=word.lower()
    set=('http','https','www','html') #Work in progress
    if word in set:
        return None
    set1={'a','i','.'}

    if word not in set1 and len(word)<2 :
        return None
    return word

def check_numbers_letters(word):
    """

    :param word: token
    :return: True if a string represent a float number or a word
            False if the string has numbers and letters combined
    """
    try:
        float(word)
        True
    except ValueError:
        if any(i.isdigit() for i in word):
            return False
        return True



def RepresentsInt(s):
    """

    :param s: token
    :return: True if the token is int
    False if the token is not int
    """
    try:
        int(s)
        return True
    except ValueError:
        return False


def default_to_regular(d):
    """

    :param d:Nested dictionary of defaultdicts
    :return: Nested dictionary of dicts
    """
    if isinstance(d, defaultdict):
        d = {k: default_to_regular(v) for k, v in d.items()}
    return d

def create_word_list(file_list):
    corpus = []
    for file in file_list:
        corpus.append(proccess_file(file))
    words = [re.findall(r"[\w']+|[\'\".,!?;#]", i) for i in corpus]
    temp = []
    for index,x in enumerate(words):
        temp += ['<END>', '<START>']
        for y in x:
            normalized_word = normalization(y)
            if normalized_word:
                temp.append(normalized_word)  # Normalizing each token see the function for the full doc

        if index+1==len(words):
            temp += ['<END>', '<START>']
    return temp

def learn_language_model(files, n=3, lm=None):
    words=create_word_list(files)
    ngrams_list = ngrams(words, n)
    ngram_dict=defaultdict(lambda: defaultdict(int))
    for grams in ngrams_list:
        ngram_dict[tuple(grams[:-1])][grams[-1]]+=1
    ngram_dict=default_to_regular(ngram_dict)

    if lm is None:
        return ngram_dict
    else:
        for k,v in lm.items():
            if k not in ngram_dict:
                ngram_dict[k]=v
            else:
                ngram_dict[k]=dict(collections.Counter(v) + collections.Counter(ngram_dict[k]))
        return ngram_dict


    return ngram_dict



def create_lexicon(files, lexicon=collections.defaultdict(int)):

    words=create_word_list(files)
    words = [x for x in words if x != '<START>' and x!='<END>']



    return_dict=dict(collections.Counter(words))
    return return_dict

    #TODO: implement merge if lexicon is given

    """Returns a dictionary of word-tokens and their counts, format: {str:int}. Counts are based on the specified files. If a lexicin is passed the counts are updated.

    Args:
        files (list): a list of files (full path) to process.
        lexicon (dict): a lexicon of word-tokens and their counts.

    """

def diff_insertion(err,corr):
    for index,i in enumerate(err):
        if index+1==len(err):
            return (err[index-1],i)
        if i!= corr[index]:
            return (corr[index-1],i) if index>0 else ('',i)

def diff_del(err, corr):
    for index,i in enumerate(corr):
        if index+1==len(corr):
            return (corr[index-1],i)
        if i!= err[index]:
            return (corr[index-1],i) if index>0 else ('',i)
def diff_sub(err,corr):
    for index,i in enumerate(corr):
        if i!= err[index]:
            return (err[index],i) if index>0 else ('',i)

def diff_trans(err,corr):
    for index,i in enumerate(corr):
        if i!= err[index]:
            return (err[index]+err[index+1],i+corr[index+1])



def create_letter_count(lexicon):
    dict_letter=defaultdict(int)
    for word,word_count in lexicon.items():
        letter_word_counter=Counter(word)
        ngram_list=[''.join(ngram) for ngram in ngrams(' '+word+' ',2)]
        for ngram in ngram_list:
            dict_letter[ngram]+=word_count
        for letter,letter_count in letter_word_counter.items():
            dict_letter[letter]+=letter_count*word_count
    return dict_letter


def create_error_distribution(errors_files, lexicon):
    corpus = []
    distribution_dict=create_letter_count(lexicon)
    total_count=sum(lexicon.values())

    for file in errors_files:
        corpus.append(proccess_file(file,'errors'))
    flat_list = [item.replace('\n','') for sublist in corpus for item in sublist]
    error_dict=defaultdict(lambda :defaultdict(float))
    for line in flat_list:
        err,corr=line.split('\t')
        err=err.replace(' ','') if err.startswith(' ') else err
        corr=corr.replace(' ','') if corr.startswith(' ') else corr

        if len(err)<len(corr):
            count=distribution_dict[''.join(diff_del(err,corr))[::-1]] if distribution_dict[''.join(diff_del(err,corr))[::-1]]>0 else total_count #TODO: better smothing
            error_dict['deletion'][diff_del(err,corr)]+=1/count
        elif len(err)>len(corr):
            count = distribution_dict[diff_insertion(err,corr)[1]] if distribution_dict[diff_insertion(err,corr)[1]] > 0 else total_count
            error_dict['insertion'][diff_insertion(corr,err)]+=1/count
        else:
            if set(err)!=set(corr):
                count = distribution_dict[diff_sub(err, corr)[1]] if distribution_dict[diff_sub(err, corr)[1]] > 0 else total_count
                error_dict['substitution'][diff_sub(err, corr)] += 1/count
            else:
                flag=True
                err_counter=Counter(err)
                corr_counter=Counter(corr)
                for k_err,v_err in err_counter.items():
                    if v_err!=corr_counter[k_err]:
                        count=distribution_dict[diff_sub(err,corr)[1]] if distribution_dict[diff_sub(err,corr)[1]]>0 else total_count
                        error_dict['substitution'][diff_sub(err, corr)] += 1/count
                        flag=False
                        break
                if flag:
                    count=distribution_dict[diff_trans(err,corr)[1]] if distribution_dict[diff_trans(err,corr)[1]]>0 else total_count
                    error_dict['transposition'][diff_trans(err, corr)] += 1/count



    return error_dict



    """ Returns a dictionary {str:dict} where str is in:
    <'deletion', 'insertion', 'transposition', 'substitution'> and the inner dict {tupple: float} represents the confution matrix of the specific errors
    where tupple is (err, corr) and the float is the probability of such an error. Examples of such tupples are ('t', 's'), for deletion of a t after an 's', insertion of a 't' after an 's'  and substitution of ;s' by a 't'; and ('ac','ca') for transposition.
    In the case of insersion  the tuppe (i,j) reads as "i was mistakingly added after j". In the casof deletion the tupple reads as "i was mistakingly ommitted after j"
    Notes:
        1. The error distributions could be represented in more efficient ways.
           We ask you to keep it simple and straight forward for clarity.
        2. Ultimately, one can use only 'deletion' and 'insertion' and have
            'substitution' and 'transposition' derived. Again,  we use all
            four explicitly in order to keep things simple.
    Args:
        errors_file (str): full path to the errors file. File format mathces
                            Wikipedia errors list.
        lexicon (dict): A dictionary of words and their counts derived from
                        the same corpus used to learn the language model.

    Returns:
        A dictionary of error distributions by error type (dict).

    """

def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                if i1+1<len(s1) and i2+1<len(s2):
                    if c1==s2[i2+1] and c2==s1[i1+1]:
                        return 1
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def get_probability(err_dist,error,func,err,corr,total_count=None,word_counts=None):
    if not total_count and not word_counts:
        return 1 if  err_dist['deletion'][func(err, corr)]==0 else err_dist['deletion'][func(err, corr)]

    channel_model =1 if  err_dist['deletion'][func(err, corr)]==0 else err_dist['deletion'][func(err, corr)]
    prior = word_counts[corr] / total_count
    return math.exp(math.log(channel_model) + math.log(prior))


def correct_word(w, word_counts, err_dist, c=1):
    word_small_distnace=[]
    for k in word_counts:
        #if levenshteinDistance(w,k)==0:
         #   return w
        if levenshteinDistance(w,k)==1:
            word_small_distnace.append(k)
    err=w
    total_count=sum(word_counts.values())
    probability_dict={}
    for corr in word_small_distnace:
        if len(err)<len(corr):
            probability_dict[corr]=get_probability(err_dist,'deletion',diff_del,err,corr,total_count,word_counts)
        elif len(err)>len(corr):
            probability_dict[corr]=get_probability(err_dist,'insertion',diff_insertion,err,corr,total_count,word_counts)
        else:
            if set(err)!=set(corr):

                probability_dict[corr] = get_probability(err_dist, 'substitution', diff_sub, err, corr, total_count,
                                                         word_counts)
            else:
                flag=True
                err_counter=Counter(err)
                corr_counter=Counter(corr)
                for k_err,v_err in err_counter.items():
                    if v_err!=corr_counter[k_err]:
                        probability_dict[corr] = get_probability(err_dist, 'substitution', diff_sub, err, corr,
                                                                 total_count,
                                                                 word_counts)
                        flag=False
                        break
                if flag:
                    probability_dict[corr] = get_probability(err_dist, 'transposition', diff_trans, err, corr, total_count,
                                                             word_counts)


    return max(probability_dict.items(), key=operator.itemgetter(1))[0]

    """ Returns the most probable correction for the specified word, given the specified prior error distribution.

    Args:
        w (str): a word to correct
        word_counts (dict): a dictionary of {str:count} containing the
                            counts  of uniqie words (from previously loaded
                             corpora).
        err_dist (dict): a dictionary of {str:dict} representing the error
                            distribution of each error type (as returned by
                            create_error_distribution() ).
        c (int): the maximal number of errors to look for (default: 1).

    Returns:
        The most probable correction (str).
    """



def get_noisy_channel(err,corr,err_dist):
    if len(err) < len(corr):
        return get_probability(err_dist, 'deletion', diff_del, err, corr)
    elif len(err) > len(corr):
        return get_probability(err_dist, 'insertion', diff_insertion, err, corr)
    else:
        if set(err) != set(corr):

            return get_probability(err_dist, 'substitution', diff_sub, err, corr)
        else:
            flag = True
            err_counter = Counter(err)
            corr_counter = Counter(corr)
            for k_err, v_err in err_counter.items():
                if v_err != corr_counter[k_err]:
                    return get_probability(err_dist, 'substitution', diff_sub, err, corr)
            return get_probability(err_dist, 'transposition', diff_trans, err, corr)


def extract_n(lm):
    """
    Helper func to extract the n-1 out of the ngram Language Model
    :param lm: Language Model
    :return: (int) n-1 of the model
    """
    ngram_counter=[]
    for ngram in lm:
        ngram_counter.append(len(ngram))
    counter=Counter(ngram_counter)
    return counter.most_common(1)[0][0]+1

def correct_sentence(s, lm, err_dist, c=1, alpha=0.95):
    corpus = [s.split(' ')]
    words = [re.findall(r"[\w']+|[\'\".,!?;#]", i) for i in corpus]
    s = []
    for index, x in enumerate(words):
        s += ['<END>', '<START>']
        for y in x:
            normalized_word = normalization(y)
            if normalized_word:
                s.append(normalized_word)  # Normalizing each token see the function for the full doc

        if index + 1 == len(words):
            s += ['<END>', '<START>']

    n=extract_n(lm)
    ngrams_list = ngrams(s, n)
    lexicon=get_lex(lm)
    possible_words=defaultdict(list)
    for err in s:
        for corr in lexicon:
            if levenshteinDistance(err, corr)<=1:
                possible_words[err].append(corr)
    distribution_dict=defaultdict(lambda: defaultdict(float))
    for err,corrs in possible_words:
        for corr in corrs:
            if err==corr:
                noisy_channel=alpha
            else:
                noisy_channel=get_noisy_channel(err,corr,err_dist)
                noisy_channel=noisy_channel*(1-alpha)/len(corrs)
            ngram=get_ngram(s,err,n)
            if ngram in lm:
                if corr in lm[ngram]:
                    lm_prob=lm[ngram][corr]/len(lm)
            else:
                lm_prob=1-len(lm)+1
            distribution_dict[err][corr]=noisy_channel*lm_prob
    max(probability_dict.items(), key=operator.itemgetter(1))
    print(2)

    """ Returns the most probable sentence given the specified sentence, language
    model, error distributions, maximal number of suumed erroneous tokens and likelihood for non-error.

    Args:
        s (str): the sentence to correct.
        lm (dict): the language model to correct the sentence accordingly.
        err_dist (dict): error distributions according to error types
                (as returned by create_error_distribution() ).
        c (int): the maximal number of tokens to change in the specified sentence.
                    (default: 1)
        alpha (float): the likelihood of a lexical entry to be the a correct word.
                    (default: 0.95)

    Returns:
        The most probable sentence (str)
    """

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



def get_lex(lm):
    lex=set()
    for ngram in lm:
        lex.update([normalization(x) for x in ngram if x != '<START>' and x!='<END>'])
    return lex