#importing the data
print('Getting Data')
train=pd.read_csv(r'C:\Users\13432\Documents\Mod 3\MNIST\mnist\mnist\mnist_train.csv')
test=pd.read_csv(r'C:\Users\13432\Documents\Mod 3\MNIST\mnist\mnist\mnist_test.csv')
#preparing the data
#breaking off the labels seperate from the data so the model doesnt know what it is
print('Preparing data')
train_label=train['label']
test_label=test['label']
test.drop(columns=['label'])
train.drop(columns=['label'])

#training the data using sk.learn
print('Training data')
# the numbers indicate the hidden layers, in this case 3 hidden layers of 100 neurons, and a hidden layer #of 5 neurons
mlpc=MLPClassifier(hidden_layer_sizes=(100,100,100,5))

#fitting the data
print('fitting data')
mlpc.fit(train,train_label)

#running the model on the test set
print('Predicting...')
prediction=mlpc.predict(test)
cm=accuracy_score(test_label,prediction)
print(cm)
#optional
#
confusion=confusion_matrix(test_label,prediction)
report=classification_report(test_label, prediction)
