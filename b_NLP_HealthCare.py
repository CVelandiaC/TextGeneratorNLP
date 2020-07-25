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
# CODE: Class created to perform all the essential tasks for language generation 
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

import numpy as np
import pandas as pd 
from nltk.tokenize import RegexpTokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, LSTM, Embedding
from keras.utils import np_utils
import random

class NLP_LanguageGen_Hc:
    """Class to load, clean, process and organise data to feed and train 
    a deeplearning algorithm to generate text."""

    def __init__(self):
        self.W_to_num = {} #Encoder        
        self.TextLength = 0 #Num of chars in TextSet
        self.vocab_len = 0 #Num of chars for encoding 
        self.N_Sequences = 0 #Num of created squences
        self.WordSet = [] #Unique Characters set (Vocabulary)

        self.model = Sequential() #ML Model 

    def textFile_Process(self, Name):
        """Load text file and return a list of words"""
        WList = pd.read_csv(Name, usecols=[0], header=None, names=['KeyWords'])
        WList = WList['KeyWords'].values.tolist()  #transform to shuffled list
        return WList

    def excelFile_Process(self, Name, Col):
        """Load excel file, take a column of sentences and return a list of words"""
        # Load excel file
        HealthData = pd.read_excel(Name)       
        HSentences = HealthData[Col]
        #Remove symbols and spaces
        HSentences = HSentences.replace(0,'0')
        HSentences = (HSentences.str.lower()).apply(RegexpTokenizer(r'[a-zA-Z]+').tokenize) 
        HSentences = HSentences.apply(lambda row: [w for w in row if len(w) >= 2]) # Drop words less than 2 chars 
        HWList = [st for row in HSentences for st in row] #Transform to huge health words list
        return HWList

    def CreateDataSet(self, WList):
        """Get all data as a concatenated list and process it to generate appropriate sets to feed
        the model, returns the processed data and X and y data sets"""

        self.WordSet = sorted(list(set(WList))) # Create set of unique words
        self.W_to_num = dict((c, i) for i, c in enumerate(self.WordSet)) #Dictionary to convert chars to numbers (Encoder)

        self.TextLength = len(WList) #Total numbeer of words
        self.vocab_len = len(self.WordSet) # Total unique words (Vocabulary)

        #------------- Define input and output data sets --------------
        Enc_x = []
        Enc_y = []
        seq = []

        for i in range(2, self.TextLength):

            x_seq = WList[i-2:i] # bigrams
            y_seq = WList[i] # Current word 

            seq.append([x_seq,y_seq]) #Save sequences for Viz
            #Encode data and add to list
            Enc_x.append([self.W_to_num[w] for w in x_seq])
            Enc_y.append(self.W_to_num[y_seq])
        
        self.N_Sequences = len(seq)

        X = np.reshape(Enc_x, (self.N_Sequences, 2, 1))

        #Convert outputs to categorical data (one hot) 
        y = np_utils.to_categorical(Enc_y)       

        return X, y, seq

    def LSTM_ModelDef(self, X, y, Units, DimProj, Dpo):
        """Define an RNN - LSTM Model with """
        #------------- LSTM MODEL definition --------------

        self.model.add(Embedding(self.vocab_len, DimProj, input_length=2)) #Input layer
        self.model.add(LSTM(Units)) #First layer
        self.model.add(Dropout(Dpo)) #Dropout to reduce overfitting
        self.model.add(Dense(self.vocab_len, activation='softmax')) # Fully connected output layer

        self.model.compile(loss='categorical_crossentropy', optimizer='adam')

    def TrainModel(self, fileName, X, y, epo, BSize):
        """Train the defined RNN - LSTM Model"""
        print("\n------------------------Training Started-------------------------")
        #Define a checkpoint to save the model weights
        checkpoint = ModelCheckpoint(fileName, monitor='loss', verbose=1,save_best_only=True,
                                    save_weights_only=False, mode='min')

        # ************** Fit the model
        self.model.fit(X, y, epochs=epo, batch_size=BSize, callbacks=[checkpoint])

    def TestModel(self, filename, TestSet, GenWords):
        """Test the RNN - LSTM Model saved at filename with the test set"""

        self.model.load_weights(filename)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')

        #Define Decoder
        num_to_word = dict((i, c) for i, c in enumerate(self.WordSet))

        FText = ["Results of Language Generation Exercise - Cristian Velandia \n 23/06/2020 \n"]

        for i in range(1,len(TestSet)):
            #Random seed from vocab + word from test set
            InitW = random.choice(list(self.W_to_num.values())) 
            W = TestSet[i]
            pattern = [InitW, self.W_to_num[W]] #Encode               
            print("Test Word: ",num_to_word[InitW], W, "/ Encoding: ", pattern)

            Rtext = num_to_word[InitW] + " " + W
            # Iteratively generate "GenWords" words 
            for _ in range(GenWords):
                
                x = np.reshape(pattern, (1, 2, 1)) #Reshape to tensor form 

                prediction = self.model.predict(x, verbose=0)
                index = np.argmax(prediction)

                pattern[0] = pattern[1]          
                pattern[1] = index

                result = num_to_word[index] 
                Rtext += ' ' + result
            print("\nResults :", Rtext, "\n")
            FText.append(Rtext + "\n")
        with open("NLPResults.txt", "w") as text_file:
            text_file.write(" ".join(FText))

       

    def ShowAllParams(self):
        print("\n----------------------------------------------------------------")
        print("Total number of characters: ", self.TextLength)
        print("Total unique characters in text: ", self.vocab_len)
        print("Total Sequences: ", self.N_Sequences)
        print("----------------------------------------------------------------")
        self.model.summary()
        print("----------------------------------------------------------------\n")

    




