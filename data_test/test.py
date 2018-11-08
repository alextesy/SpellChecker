from assignment2 import *
import datetime
data_folder = ""#"data_test\\"


time=datetime.datetime.now()
list=['big.txt']
error_list=['commmon_errors.txt']

list=[data_folder+i for i in list]
lexicon = create_lexicon(list)
alex=create_error_distribution(error_list,lexicon)
#print(evaluate_text('once upon a midnight dreary',3,lm))
print(datetime.datetime.now()-time)
