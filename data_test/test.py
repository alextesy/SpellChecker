from assignment2 import *
import datetime
data_folder = "data_test\\"


time=datetime.datetime.now()
list=['big.txt','trump_historical_tweets.txt']
error_list=['commmon_errors.txt']
#list=[data_folder+i for i in list]

lm=learn_language_model(list)
print('lm '  +str(datetime.datetime.now()-time))
time=datetime.datetime.now()

lexicon=create_lexicon(list)
print('lexicon '  +str(datetime.datetime.now()-time))
time=datetime.datetime.now()


error_dist=create_error_distribution(error_list,lexicon)
print('error dist '  +str(datetime.datetime.now()-time))
time=datetime.datetime.now()



print(correct_sentence('when will your time correctly however',lm,error_dist))

print('correct sent'  +str(datetime.datetime.now()-time))
time=datetime.datetime.now()
'''print(correct_word('abreviated',lexicon,error_dist))
print(correct_word('asfdavg',lexicon,error_dist))
print(correct_word('abreviated',lexicon,error_dist))
print(correct_word('abreviated',lexicon,error_dist))
print(correct_word('accomodations',lexicon,error_dist))
print(correct_word('acadamy',lexicon,error_dist))
print(correct_word('absail',lexicon,error_dist))
print(correct_word('abreviated',lexicon,error_dist))
print(correct_word('persuing',lexicon,error_dist))
'''


