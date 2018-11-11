from assignment2 import *
import datetime
data_folder = "data_test\\"


time=datetime.datetime.now()
list=['big.txt','trump_historical_tweets.txt']
error_list=['commmon_errors.txt']
#list=[data_folder+i for i in list]
lm=learn_language_model(list,n=3)

lexicon=create_lexicon(list)
error_dist=create_error_distribution(error_list,lexicon)
corr_sent=correct_sentence('Be so good as to step in',lm,error_dist)
print(corr_sent)
print(datetime.datetime.now()-time)
