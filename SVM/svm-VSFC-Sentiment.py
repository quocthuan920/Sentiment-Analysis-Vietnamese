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

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV 
from sklearn.pipeline import Pipeline

# save and load a file
import pickle

df_train=pd.DataFrame(list(zip(open('data/_UIT-VSFC/train/sents.txt', 'r',encoding="utf8"), open('data/_UIT-VSFC/train/sentiments.txt', 'r',encoding="utf8")))) 
df_test=pd.DataFrame(list(zip(open('data/_UIT-VSFC/dev/sents.txt', 'r',encoding="utf8"), open('data/_UIT-VSFC/dev/sentiments.txt', 'r',encoding="utf8"))))
df_train.columns =['Sents', 'Sentiments']
df_test.columns =['Sents', 'Sentiments']

df_train=df_train.replace("\n","", regex=True)
df_test=df_test.replace("\n","", regex=True)


X_train = df_train.Sents
X_test = df_test.Sents

y_train = df_train.Sentiments
y_test = df_test.Sentiments

class_names = ['0', '1', '2']
data = pd.concat([df_train, df_test])
#print(data)
print('Kích thước tập train: %s' % (len(df_train['Sents'])))
print('Kích thước tập test: %s' % (len(df_test['Sents'])))
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

    # Chuyển dấu câu thành khoảng trắng
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
    
    #remove khoảng trắng dư thừa ở đầu và cuối chuỗi
    data = data.strip()
    #tách từ
    data = ViTokenizer.tokenize(data)
    data = data.split()
    data = [t.replace('_', ' ') for t in data]
    return data


# TFIDF, unigrams and bigrams
vect = TfidfVectorizer( sublinear_tf=True, norm='l2', ngram_range=(1, 2))

# fit on our complete corpus
vect.fit_transform(data.Sents)

# transform testing and training datasets to vectors
X_train_vect = vect.transform(X_train)
X_test_vect = vect.transform(X_test)
'''
param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}   
  
svc = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3) 
  
# fitting the model for grid search 
svc.fit(X_train_vect, y_train) 
print(svc.best_params_) 
# print how our model looks after hyper-parameter tuning 
print(svc.best_estimator_) 

ysvm_pred = svc.predict(X_test_vect) 
# print classification report 
print(classification_report(y_test, ysvm_pred, digits=4)) 
'''
#svm
svc = LinearSVC(tol=1e-05)
svc.fit(X_train_vect, y_train)

ysvm_pred = svc.predict(X_test_vect)
report1 = metrics.classification_report(y_test, ysvm_pred, labels=['0', '1', '2'], digits=3)
print('Report: ',report1)



print("\nConfusion Matrix:\n", confusion_matrix(y_test, ysvm_pred))
print("\nAccuracy: {:.2f}%".format(accuracy_score(y_test, ysvm_pred) * 100))
print("Weighted F1 Score: {:.2f}%".format(f1_score(y_test, ysvm_pred, average='weighted') * 100))

plot_confusion_matrix(y_test, ysvm_pred, classes=class_names, normalize=False, title='Confusion matrix')
plt.show()

#Create pipeline with our tf-idf vectorizer and LinearSVC model
svm_model = Pipeline([
    ('tfidf', vect),
    ('clf', svc),
])

# save the model to disk

filename = 'models/svm_model_VSFC_Sentiments.sav'
pickle.dump(svm_model, open(filename, 'wb'))
 
# some time later...
 
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)

# #SAVE FILE SUBMIT
input_test = pd.DataFrame(list(zip(open('data/_UIT-VSFC/test/sents.txt', 'r',encoding="utf8"), open('data/_UIT-VSFC/test/sentiments.txt', 'r',encoding="utf8")))) 
input_test.columns =['Sents', 'Sentiments']
test_list = []
for document in input_test.Sents:
    test_list.append(document)
z_predict = loaded_model.predict(['thầy dạy dễ hiểu','máy chiếu hơi mờ','môn này khó quá'])
print('Dự đoán sentiment: ', z_predict)
y_predict = loaded_model.predict(test_list)
input_test['Sentiments'] = y_predict
# test_data['content'] = test_list
input_test = df_test.sort_values(by=['Sentiments'])
input_test[['Sents', 'Sentiments']].to_csv('submits/svm_VSFC_Sentiments.csv',encoding="utf-8-sig", index=False)
