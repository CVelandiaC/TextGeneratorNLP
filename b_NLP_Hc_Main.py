# Text Generation for Predictvia ML Specialist Job
# Version 0.0 - Code Creation, Char prediction 19/06/2020
# Version 0.1 - Code reorganization (OOP) 20/06/2020
# Version 0.2 - Adaptation for word prediction 22/06/2020
# by Cristian C. Velandia C. 
# Virtual environment at C:\Windows\system32\venvNLP\Scripts\ (Activate-Deactivate)
# Sources:
# https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-language-model-nlp-python-code/
# https://stackabuse.com/text-generation-with-python-and-tensorflow-keras/
# https://machinelearningmastery.com/develop-word-based-neural-language-models-python-keras/
# All codes were implemented, read, analyzed and understood to finally create this code by using their 
# strenghts combined with my little experience and knowledge. Also, the algorithms are improved using 
# bi-grams as input and single word output. The code use pandas for easy data loading
# and processing, then a deep learning model using embeeding input, LSTM hidden layer and fully (Dense)
# connected output is trained and tuned using the loss metric.
#
# CODE: Main code for Language generation, uses the class at b_NLP_HealthCare.py train an test can be 
# done here saving time and space. Also, improves tuning. 
#
# Packages: Python 3.8.3 
# h5py                     2.10.0
# Keras                    2.4.2
# nltk                     3.5
# numpy                    1.18.5
# pandas                   1.0.5
# scikit-learn             0.23.1
# scipy                    1.4.1
# tensorflow               2.2.0

from b_NLP_HealthCare import NLP_LanguageGen_Hc

LGen = NLP_LanguageGen_Hc() #Create a new obj for training

#Load Data
TrainKW = LGen.textFile_Process('keywords.txt')
TestKW = LGen.textFile_Process('keywordsTEST.txt')

HealthData = LGen.excelFile_Process('health_claim_data_submit.xls','news_title')

#Merge string lists
Ds = TrainKW + HealthData + TestKW 

#Generate datasets using monograms
X, y, seq = LGen.CreateDataSet(Ds)
#print(seq)

#Define Model
LGen.LSTM_ModelDef(X, y, Units=512, DimProj=30, Dpo=0.3)

#Show all parameters
LGen.ShowAllParams()

Train = 1 #Mode Selection
if Train == 0:
    #TRAIN MODEL
    LGen.TrainModel("NLP_model_weights_Words.hdf5", X, y, epo=200, BSize=1024)
else:
    #TEST MODEL
    LGen.TestModel("NLP_model_weights_Words.hdf5", TestKW, GenWords=8)