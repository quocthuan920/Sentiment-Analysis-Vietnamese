# -*- coding: utf-8 -*-
"""Bert_ViCTSD_Toxicity.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1IPIb47RPZMdjcnfeoUDAlkIulTdvEELg

# BERT


## Workflow: 
1. Import Data
2. Data preprocessing and downloading BERT
3. Training and validation
4. Saving the model

Multiclass text classification with BERT and [ktrain](https://github.com/amaiya/ktrain). Use google colab for a free GPU 

👋  **Let's start**
"""

# install ktrain on Google Colab
!pip3 install ktrain

import pandas as pd
import numpy as np

import ktrain
from ktrain import text

"""## 1. Import Data

Thay data_train.csv thành train_VMSEC.csv
Thay data_test.csv thành valid_VMSEC.csv
Thay Text thành Sentence
"""

data_train = pd.read_csv('/content/data/UIT-ViCTSD_train.csv', encoding='utf-8') 
data_test = pd.read_csv('/content/data/UIT-ViCTSD_valid.csv', encoding='utf-8')

X_train = data_train.Comment.tolist()
X_test = data_test.Comment.tolist()

y_train = data_train.Toxicity.tolist()
y_test = data_test.Toxicity.tolist()

data = data_train.append(data_test, ignore_index=True)

class_names = ['0', '1']

print('size of training set: %s' % (len(data_train['Comment'])))
print('size of validation set: %s' % (len(data_test['Comment'])))
print(data.Toxicity.value_counts())

data.head(10)



"""## 2. Data preprocessing

* The text must be preprocessed in a specific way for use with BERT. This is accomplished by setting preprocess_mode to ‘bert’. The BERT model and vocabulary will be automatically downloaded

* BERT can handle a maximum length of 512, but let's use less to reduce memory and improve speed. 
"""

(x_train,  y_train), (x_test, y_test), preproc = text.texts_from_array(x_train=X_train, y_train=y_train,
                                                                       x_test=X_test, y_test=y_test,
                                                                       class_names=class_names,
                                                                       preprocess_mode='bert',
                                                                       maxlen=350, 
                                                                       max_features=35000)

"""## 2. Training and validation

Loading the pretrained BERT for text classification
"""

model = text.text_classifier('bert', train_data=(x_train, y_train), preproc=preproc)

"""Wrap it in a Learner object"""

learner = ktrain.get_learner(model, train_data=(x_train, y_train), 
                             val_data=(x_test, y_test),
                             batch_size=6)

"""Train the model. More about tuning learning rates [here](https://github.com/amaiya/ktrain/blob/master/tutorial-02-tuning-learning-rates.ipynb)"""

learner.fit_onecycle(2e-5, 3)

"""Validation"""

learner.validate(val_data=(x_test, y_test), class_names=class_names)

"""#### Testing with other inputs"""

predictor = ktrain.get_predictor(learner.model, preproc)
predictor.get_classes()

import time 

message = 'điều này thật là tốt'

start_time = time.time() 
prediction = predictor.predict(message)

print('predicted: {} ({:.2f})'.format(prediction, (time.time() - start_time)))

"""## 4. Saving Bert model

"""

# let's save the predictor for later use
predictor.save("models/bert_model_VMSEC")

"""Done! to reload the predictor use: ktrain.load_predictor"""