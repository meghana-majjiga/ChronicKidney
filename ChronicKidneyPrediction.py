from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import os
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint 
import pickle
from keras.layers import LSTM
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential, Model
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import seaborn as sns



main = tkinter.Tk()
main.title("Chronic Kidney Disease Prediction using CNN, LSTM & Ensemble Model") 
main.geometry("1300x1200")

global dataset, X, Y, X_train, y_train, X_test, y_test, cnn_model
global accuracy, precision, recall, fscore, labels
global label_encoder, scaler

def loadData():
    global dataset, labels
    filename = filedialog.askopenfilename(initialdir="Dataset")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+" dataset loaded\n\n")
    dataset = pd.read_csv(filename)
    text.insert(END,str(dataset.head()))
    labels = np.unique(dataset['classification'])

    label = dataset.groupby('classification').size()
    label.plot(kind="bar")
    plt.xlabel("Chronic Kidney Disease Type")
    plt.ylabel("Count")
    plt.title("Chronic Kidney Disease Graph")
    plt.show()
    
def datasetProcessing():
    text.delete('1.0', END)
    global dataset, label_encoder, scaler, X, Y, X_train, y_train, X_test, y_test
    dataset.fillna(0, inplace = True)

    label_encoder = []
    columns = dataset.columns
    types = dataset.dtypes.values
    for i in range(len(types)):
        name = types[i]
        if name == 'object': #finding column with object type
            le = LabelEncoder()
            dataset[columns[i]] = pd.Series(le.fit_transform(dataset[columns[i]].astype(str)))#encode all str columns to numeric 
            label_encoder.append(le)
    text.insert(END,"Dataset after preprocessing\n\n")
    text.insert(END,str(dataset)+"\n\n")
    dataset = dataset.values
    X = dataset[:,1:dataset.shape[1]-1]
    Y = dataset[:,dataset.shape[1]-1]
    Y = Y.astype(int)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    Y = to_categorical(Y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1, 1))

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Dataset Train & Test Splits\n")
    text.insert(END,"Total records found in dataset : "+str(X.shape[0])+"\n")
    text.insert(END,"80% dataset used for training  : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% dataset user for testing   : "+str(X_test.shape[0])+"\n")

def calculateMetrics(algorithm, testY, predict):
    global labels
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Accuracy  : "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FSCORE    : "+str(f)+"\n\n")
    conf_matrix = confusion_matrix(testY, predict)
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()        

def runCNN():
    text.delete('1.0', END)
    global accuracy, precision, recall, fscore, cnn_model
    global X_train, y_train, X_test, y_test
    accuracy = []
    precision = []
    recall = [] 
    fscore = []

    cnn_model = Sequential()
    #adding CNN layer wit 32 filters to optimized dataset features using 32 neurons
    cnn_model.add(Convolution2D(32, (1, 1), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
    #adding maxpooling layer to collect filtered relevant features from previous CNN layer
    cnn_model.add(MaxPooling2D(pool_size = (1, 1)))
    #adding another CNN layer to further filtered features
    cnn_model.add(Convolution2D(32, (1, 1), activation = 'relu'))
    cnn_model.add(MaxPooling2D(pool_size = (1, 1)))
    #collect relevant filtered features
    cnn_model.add(Flatten())
    #defining output layers
    cnn_model.add(Dense(units = 256, activation = 'relu'))
    #defining prediction layer with Y target data
    cnn_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
    #compile the CNN with LSTM model
    cnn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    #train and load the model
    if os.path.exists("model/cnn_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/cnn_weights.hdf5', verbose = 1, save_best_only = True)
        hist = cnn_model.fit(X_train, y_train, batch_size = 8, epochs = 1, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/cnn_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        cnn_model.load_weights("model/cnn_weights.hdf5")
    predict = cnn_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    testY = np.argmax(y_test, axis=1)
    calculateMetrics("CNN", testY, predict)
    
def runLSTM():
    global X_train, y_train, X_test, y_test
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2] * X_train.shape[3]))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2] * X_test.shape[3]))
    lstm_model = Sequential()#defining deep learning sequential object
    #adding LSTM layer with 100 filters to filter given input X train data to select relevant features
    lstm_model.add(LSTM(100,input_shape=(X_train.shape[1], X_train.shape[2])))
    #adding dropout layer to remove irrelevant features
    lstm_model.add(Dropout(0.5))
    #adding another layer
    lstm_model.add(Dense(100, activation='relu'))
    #defining output layer for prediction
    lstm_model.add(Dense(y_train.shape[1], activation='softmax'))
    #compile LSTM model
    lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #start training model on train data and perform validation on test data
    #train and load the model
    if os.path.exists("model/lstm_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/lstm_weights.hdf5', verbose = 1, save_best_only = True)
        hist = lstm_model.fit(X_train, y_train, batch_size = 8, epochs = 10, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/lstm_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        lstm_model.load_weights("model/lstm_weights.hdf5")
    predict = lstm_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    testY = np.argmax(y_test, axis=1)
    calculateMetrics("LSTM", testY, predict)

def runEnsemble():
    global X_train, y_train, X_test, y_test, cnn_model, Y
    ensemble_model = Model(cnn_model.inputs, cnn_model.layers[-2].output)#creating cnn model
    cnn_features = ensemble_model.predict(X)  #extracting cnn features from test data
    Y = np.argmax(Y, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(cnn_features, Y, test_size=0.2)

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    predict = rf.predict(X_test)
    calculateMetrics("Ensemble CNN with Random Forest", y_test, predict)
    

def graph():
    df = pd.DataFrame([['CNN','Accuracy',accuracy[0]],['CNN','Precision',precision[0]],['CNN','Recall',recall[0]],['CNN','FSCORE',fscore[0]],
                       ['LSTM','Accuracy',accuracy[1]],['LSTM','Precision',precision[1]],['LSTM','Recall',recall[1]],['LSTM','FSCORE',fscore[1]],
                       ['Ensemble CNN with Random Forest','Accuracy',accuracy[2]],['Ensemble CNN with Random Forest','Precision',precision[2]],['Ensemble CNN with Random Forest','Recall',recall[2]],['Ensemble CNN with Random Forest','FSCORE',fscore[2]],
                      ],columns=['Algorithms','Accuracy','Value'])
    df.pivot("Algorithms", "Accuracy", "Value").plot(kind='bar')
    plt.title("All Algorithm Comparison Graph")
    plt.show()

def predictDisease():
    text.delete('1.0', END)
    global label_encoder, scaler, cnn_model, labels
    filename = filedialog.askopenfilename(initialdir="Dataset")
    dataset = pd.read_csv(filename)
    dataset.fillna(0, inplace = True)
    temp = dataset.values
    index = 0
    columns = dataset.columns
    types = dataset.dtypes.values
    for i in range(len(types)):
        name = types[i]
        if name == 'object': #finding column with object type
            dataset[columns[i]] = pd.Series(label_encoder[index].transform(dataset[columns[i]].astype(str)))#encode all str columns to numeric 
            index = index + 1
    dataset = dataset.values
    dataset = dataset[:,1:dataset.shape[1]]
    X = scaler.transform(dataset)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1, 1))
    predict = cnn_model.predict(X)
    for i in range(len(predict)):
        pred = np.argmax(predict[i])
        text.insert(END,"Test Data = "+str(temp[i])+" =====> Predicted As "+str(labels[pred])+"\n\n")
    

font = ('times', 16, 'bold')
title = Label(main, text='Chronic Kidney Disease Prediction using CNN, LSTM & Ensemble Model')
title.config(bg='brown', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Chronic Kidney Dataset", command=loadData)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=460,y=100)

preprocessButton = Button(main, text="Preprocess Dataset", command=datasetProcessing)
preprocessButton.place(x=50,y=150)
preprocessButton.config(font=font1) 

cnnButton = Button(main, text="Run CNN Algorithm", command=runCNN)
cnnButton.place(x=330,y=150)
cnnButton.config(font=font1) 

lstmButton = Button(main, text="Run LSTM Algorithm", command=runLSTM)
lstmButton.place(x=630,y=150)
lstmButton.config(font=font1)

ensembleButton = Button(main, text="Run Ensemble Random Forest", command=runEnsemble)
ensembleButton.place(x=50,y=200)
ensembleButton.config(font=font1) 

graphButton = Button(main, text="Comparison Graph", command=graph)
graphButton.place(x=330,y=200)
graphButton.config(font=font1)

predictButton = Button(main, text="Predict Disease from Test Data", command=predictDisease)
predictButton.place(x=630,y=200)
predictButton.config(font=font1) 


font1 = ('times', 12, 'bold')
text=Text(main,height=22,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1)


main.config(bg='brown')
main.mainloop()
