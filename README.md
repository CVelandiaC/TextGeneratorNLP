# TextGeneratorNLP
NLP: Healthcare text generator using Tensorflow, keras LSTM network with embedding input and fully connected output. 

by Cristian C. Velandia C. 

Sources:
https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-language-model-nlp-python-code/
https://stackabuse.com/text-generation-with-python-and-tensorflow-keras/
https://machinelearningmastery.com/develop-word-based-neural-language-models-python-keras/

All codes were implemented, read, analyzed and understood to finally create this code by using their 
strenghts combined with my little experience and knowledge. Also, the algorithms are improved using 
bi-grams as input and single word output. The code use pandas for easy data loading
and processing, then a deep learning model using embeeding input, LSTM hidden layer and fully (Dense)
connected output is trained and tuned using the loss metric.

CODE: Class created to perform all the essential tasks for language generation 

Packages: Python 3.8.3 
h5py                     2.10.0
Keras                    2.4.2
nltk                     3.5
numpy                    1.18.5
pandas                   1.0.5
scikit-learn             0.23.1
scipy                    1.4.1
tensorflow               2.2.0
