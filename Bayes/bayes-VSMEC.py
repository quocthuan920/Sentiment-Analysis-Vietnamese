import pandas as pd
import numpy as np
# text preprocessing
from pyvi import ViTokenizer
import string
import re

# plots and metrics
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report

# feature extraction / vectorization
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# classifiers
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline

# save and load a file
import pickle

df_train = pd.read_csv('data/UIT-VSMEC/train_nor_811.csv')
df_test = pd.read_csv('data/UIT-VSMEC/valid_nor_811.csv')

X_train = df_train.Sentence
X_test = df_test.Sentence

y_train = df_train.Emotion
y_test = df_test.Emotion

class_names = ['Other', 'Disgust', 'Enjoyment', 'Surprise', 'Sadness', 'Anger', 'Fear']
data = pd.concat([df_train, df_test])

print('Kích thước tập train: %s' % (len(df_train['Sentence'])))
print('Kích thước tập test: %s' % (len(df_test['Sentence'])))
print(data.Emotion.value_counts())
data.head()

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    '''
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    '''
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    
    # Set size
    fig.set_size_inches(12.5, 7.5)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.grid(False)
    
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
    
def preprocess_and_tokenize(data):   
 
    
    #Remove các ký tự kéo dài: vd: đẹppppppp
    data = re.sub(r'([A-Z])\1+', lambda m: m.group(1).upper(), data, flags=re.IGNORECASE)
    
    # Chuyển thành chữ thường
    data = data.lower()

    #remove urls
    data = re.sub(r'http\S+', '', data)
    
    #remove hashtags và tag
    data= re.sub(r"(#[\d\w\.]+)", '', data)
    data= re.sub(r"(@[\d\w\.]+)", '', data)
    
    # chuyen punctuation thành space
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    data = data.translate(translator)
    
    #Xử lý emoj
    replace_list = {
        #Quy các icon về 7 loại emoj:'Disgust', 'Enjoyment', 'Surprise', 'Sadness', 'Anger', 'Fear'
        ":))": " Enjoyment ",":)))": " Enjoyment ","😄": " Enjoyment ", "😆": " Enjoyment ", "😂": " Enjoyment ",'🤣': ' Enjoyment ', '😊': ' Enjoyment ',
        "😍": " Enjoyment ", "😘": " Enjoyment ", "😗": " Enjoyment ","😙": " Enjoyment ", "😚": " Enjoyment ", "🤗": " Enjoyment ",
        "😇": " Enjoyment ", "😝": " Enjoyment ",  "😋": " Enjoyment ","💕": " Enjoyment ", "🧡": " Enjoyment ",'💞':' Enjoyment ',
        '💓': ' Enjoyment ', '💗': ' Enjoyment ','👍': ' Enjoyment ', '❣': ' Enjoyment ','☀': ' Enjoyment ',
        '😳': ' Surprise ', '😲': ' Surprise ', '😯': ' Surprise ', '😣': ' Sadness Fear ',
        '😢': ' Sadness Fear ', '😢': ' Sadness Fear ', '😭': ' Sadness Fear ', '😟': ' Sadness Fear ', '😢': ' Sadness Fear ',
        '😓': ' Sadness Fear ', '😞': ' Sadness ', '😔': ' Sadness ', '☹️': ' Sadness ', ':((': ' Sadness ',
        ':(((': ' Sadness ', '🙁': ' Sadness ', '😤': ' Anger ', '😠': ' Anger ', '😡': ' Anger ',
        '😒': ' Anger ', '😨': ' Disgust ', '🤢': ' Disgust ', '😧': ' Surprise ',}

    for k, v in replace_list.items():
        data = data.replace(k, v)
    
    #remove khoảng trắng
    data = data.strip()
    
    #tách từ
    data = ViTokenizer.tokenize(data)
    data = data.split()
    len_text = len(data)

    data = [t.replace('_', ' ') for t in data]
   
    return data
# TFIDF, unigrams and bigrams
vect = TfidfVectorizer(tokenizer=preprocess_and_tokenize, sublinear_tf=True, norm='l2', ngram_range=(1, 2))
print(data)
# fit on our complete corpus
vect.fit_transform(data.Sentence)

# transform testing and training datasets to vectors
X_train_vect = vect.transform(X_train)
X_test_vect = vect.transform(X_test)


#BAYES
nb = MultinomialNB()

nb.fit(X_train_vect, y_train)

ynb_pred = nb.predict(X_test_vect)
report1 = metrics.classification_report(y_test, ynb_pred, labels=['Other', 'Disgust', 'Enjoyment', 'Surprise', 'Sadness', 'Anger', 'Fear'], digits=3, zero_division=0)
print('Report: ',report1)

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

print("Accuracy: {:.2f}%".format(accuracy_score(y_test, ynb_pred) * 100))
print("\nMacro F1 Score: {:.2f}".format(f1_score(y_test, ynb_pred, average='macro') * 100))
print("\nCOnfusion Matrix:\n", confusion_matrix(y_test, ynb_pred))

# Plot normalized confusion matrix
plot_confusion_matrix(y_test, ynb_pred, classes=class_names, normalize=True, title='Normalized confusion matrix')
plt.show()

#Create pipeline with our tf-idf vectorizer and LinearSVC model
BAYES_model = Pipeline([
    ('tfidf', vect),
    ('clf', nb),
])

# save the model to disk

filename = 'models/BAYES_model_VSMEC.sav'
pickle.dump(BAYES_model, open(filename, 'wb'))
 
# some time later...
 
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)

# #SAVE FILE SUBMIT
input_test = pd.read_csv('data/UIT-VSMEC/test_nor_811.csv')

test_list = []
for document in input_test.Sentence:
    test_list.append(document)
y_predict = loaded_model.predict(test_list)
input_test['Emotion'] = y_predict
# test_data['content'] = test_list
input_test = df_test.sort_values(by=['Emotion'])
input_test[['Sentence', 'Emotion']].to_csv('submits/BAYES_VSMEC_submit.csv',encoding="utf-8-sig", index=False)

