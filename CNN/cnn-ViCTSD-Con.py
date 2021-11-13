import pandas as pd
import numpy as np

# text preprocessing
#from nltk.tokenize import word_tokenize
from pyvi import ViTokenizer
import re

# plots and metrics
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# preparing input to our model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# keras layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense

# Number of labels:0, 1
num_classes = 2

# Number of dimensions for word embedding
embed_num_dims = 300

# Max input length (max number of words) 
max_seq_len = 500

class_names = [0,1]

data_train = pd.read_csv('data/UIT-ViCTSD_train.csv', encoding='utf-8')
data_test = pd.read_csv('data/UIT-ViCTSD_valid.csv', encoding='utf-8')

X_train = data_train.Comment
X_test = data_test.Comment

y_train = data_train.Constructiveness
y_test = data_test.Constructiveness

data = data_train.append(data_test, ignore_index=True)
def preprocess_and_tokenize(data):   
 
    #remove html markup
    data = re.sub("(<.*?>)", "", data)
    
    
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
    translator = str.maketrans(str.punctuation, ' ' * len(str.punctuation))
    data = data.translate(translator)
       
    #remove khoảng trắng
    data = data.strip()
    
    #tách từ
    data = ViTokenizer.tokenize(data)
    data = data.split()
    len_text = len(data)

    data = [t.replace('_', ' ') for t in data]
    
    
    return data
print(data.Constructiveness.value_counts())
data.head(6)

def clean_text(data):
    
    # remove hashtags and @usernames
    data = re.sub(r"(#[\d\w\.]+)", '', data)
    data = re.sub(r"(@[\d\w\.]+)", '', data)
    
    # tekenization using nltk
    # data = word_tokenize(data)
    data = ViTokenizer.tokenize(data)
    return data

texts = [' '.join(clean_text(text)) for text in data.Comment]

texts_train = [' '.join(clean_text(text)) for text in X_train]
texts_test = [' '.join(clean_text(text)) for text in X_test]

print(texts_train[92])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

sequence_train = tokenizer.texts_to_sequences(texts_train)
sequence_test = tokenizer.texts_to_sequences(texts_test)

index_of_words = tokenizer.word_index

# vacab size is number of unique words + reserved 0 index for padding
vocab_size = len(index_of_words) + 1

print('Number of unique words: {}'.format(len(index_of_words)))

X_train_pad = pad_sequences(sequence_train, maxlen = max_seq_len )
X_test_pad = pad_sequences(sequence_test, maxlen = max_seq_len )

X_train_pad



# Integer labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

y_train

def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    with open(filepath, encoding="utf8") as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word] 
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]
    return embedding_matrix

import urllib.request
import zipfile
import os

fname = 'embeddings/wiki-news-300d-1M.vec'

if not os.path.isfile(fname):
    print('Downloading word vectors...')
    urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip',
                              'wiki-news-300d-1M.vec.zip')
    print('Unzipping...')
    with zipfile.ZipFile('wiki-news-300d-1M.vec.zip', 'r') as zip_ref:
        zip_ref.extractall('embeddings')
    print('done.')
    
    os.remove('wiki-news-300d-1M.vec.zip')

embedd_matrix = create_embedding_matrix(fname, index_of_words, embed_num_dims)
embedd_matrix.shape

# Inspect unseen words
new_words = 0

for word in index_of_words:
    entry = embedd_matrix[index_of_words[word]]
    if all(v == 0 for v in entry):
        new_words = new_words + 1

print('Words found in wiki vocab: ' + str(len(index_of_words) - new_words))
print('New words found: ' + str(new_words))

# Embedding layer before the actaul BLSTM 
embedd_layer = Embedding(vocab_size,
                         embed_num_dims,
                         input_length = max_seq_len,
                         weights = [embedd_matrix],
                         trainable=False)

# Convolution
kernel_size = 3
filters = 256

model = Sequential()
model.add(embedd_layer)
model.add(Conv1D(filters, kernel_size, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(256, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.summary()

batch_size = 256
epochs = 6

hist = model.fit(X_train_pad, y_train, 
                 batch_size=batch_size,
                 epochs=epochs,
                 validation_data=(X_test_pad,y_test))

# Accuracy plot
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# Loss plot
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

predictions = model.predict(X_test_pad)
predictions = np.argmax(predictions, axis=1)
predictions = [class_names[pred] for pred in predictions]

print("Accuracy: {:.2f}%".format(accuracy_score(data_test.Constructiveness, predictions) * 100))
print("\nMicro F1 Score: {:.2f}".format(f1_score(data_test.Constructiveness, predictions, average='micro') * 100))
print("\nMacro F1 Score: {:.2f}".format(f1_score(data_test.Constructiveness, predictions, average='macro') * 100))
print("\nWeighted F1 Score: {:.2f}".format(f1_score(data_test.Constructiveness, predictions, average='weighted') * 100))


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

    
print('Message: {}\nPredicted: {}'.format(X_test[4], predictions[4]))

import time

message = ['Thật là tuyệt vời']

seq = tokenizer.texts_to_sequences(message)
padded = pad_sequences(seq, maxlen=max_seq_len)

start_time = time.time()
pred = model.predict(padded)

print('Message: ' + str(message))
print('predicted: {} ({:.2f} seconds)'.format(class_names[np.argmax(pred)], (time.time() - start_time)))

# creates a HDF5 file 'my_model.h5'
model.save('models/cnn_ViCTSD_Constructiveness.h5')

from keras.models import load_model
predictor = load_model('models/cnn_ViCTSD_Constructiveness.h5')

# load the model from disk
loaded_model = load_model


# #SAVE FILE SUBMIT
input_test = pd.read_csv('data/datasets/UIT-ViCTSD/UIT-ViCTSD_test_text.csv')

test_list = []
for document in input_test.Comment:
    test_list.append(document)
predictor = load_model('models/cnn_ViCTSD_Constructiveness.h5')
input_test['Constructiveness'] = predictor
# test_data['content'] = test_list
input_test = data_test.sort_values(by=['Constructiveness'])
input_test[['Comment', 'Constructiveness']].to_csv('submits/cnn_ViCTSD_Constructiveness.csv',encoding="utf-8-sig", index=False)