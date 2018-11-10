from assignment2 import *
import datetime
data_folder = ""#"data_test\\"


time=datetime.datetime.now()
list=['big.txt']
error_list=['commmon_errors.txt']
lm=learn_language_model(list,n=3)
list=[data_folder+i for i in list]
error_dist=create_error_distribution(error_list,lexicon)
corr_sent=correct_sentence('rubbed his eyese',lm,error_dist)
print(datetime.datetime.now()-time)
