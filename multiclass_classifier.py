print("Import libraries")
import os
from nltk.corpus import stopwords
import io
from stemming.porter2 import stem
from textblob import TextBlob as tb
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from textblob.classifiers import NaiveBayesClassifier
import _pickle as cPickle
import time



print("Reading input file")
# Setting the path & filename
os.chdir('I:\\Tests\\Data-Scientist-master\\')
dataset_path = "I:\\Tests\\Data-Scientist-master\\"
filename = "trainingdataOrig.txt"


## returns term-frequency for the word
def tf(word, blob):
    return blob.words.count(word) / len(blob.words)

## returns number of unique words for the word
def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob)

## returns inverse document frequency for the word
def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

## returns term-frequency-inverse-document-frequency for the word
def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)

## stopwords from nltk corpus
stop = stopwords.words("english")


print("Initial preprocessing...")

## reads the file, ignore the first line
## remove stopwords
## creates list of document and respective class
category_list,docword_list = [],[]
with io.open(dataset_path+filename, encoding='utf-8') as file:
    for i,line in enumerate(file):
        if (i == 0):
            continue; 
        data = ' '.join([stem(word) for word in line.split() if word not in stop])
        category_list.append(data.split(' ', 1)[0])
        docword_list.append(tb(data.split(' ', 1)[1]))


print("removing words based on tfidf...")

## removes words from each document whose tfidf < 0.0
## stores trimmed corpus in a dataframe
document_list = list()
for i, blob in enumerate(docword_list):
    word_list = list()
    scores = {word: tfidf(word, blob, docword_list) for word in blob.words}
    sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for word, score in sorted_words:#[:100]:
        if score>0.0:
            word_list.append(word)   
    document_list.append(' '.join(word_list))        
    
document = pd.DataFrame({'Document':document_list,'category':category_list})



## splits dataset into train and test with 80/20 ratio.
train, test = train_test_split(document, test_size = 0.2)


## extracts class frequency in desending order    
class_freq = train['category'].value_counts().to_frame()
class_freq['index'] = class_freq.index

## gets a set of unique classes 
distinct_classes = set(train['category'])




## fucntion creates classifier to each class seperately
## so every classifier acts as a binary classifier 
## this helps to overcome the issues of imbalanced classes dataset
## first it trains classifier between class '1' and all others,
## which distributes to around 50/50 in dataset.
## Next, removes all documents with class '1' 
## and train classifier for next major class and so on.
def create_classifier(doc,class_freq):
        
    ## iterates through all classes in dataset
    for i in class_freq: 
        print('\nIteration:',i,' Category:',class_freq[i])
        ## create a copy of dataframe
        temp_doc = doc.copy()
        ## assign class as '-1' to all other classes.
        temp_doc.category[temp_doc['category'] != class_freq[i]] = '-1'
    
        temp_doc = temp_doc.values.tolist()
        ## training classifier on the temp_doc
        classifier = NaiveBayesClassifier(temp_doc)
        
        # save the classifier on disk
        with open('classifier_'+i+'.pkl', 'wb') as fid:
            cPickle.dump(classifier, fid)
        ## reassign doc with the new reduced dataset
        doc = doc[doc['category'] != class_freq[i]].copy()
      
## to track time taken in creating classifiers
print("Creating classifiers")
start = time.time()
## Training Models and Saving
create_classifier(train.copy(),class_freq['index'])
end = time.time()
TrainingSavingModel = end - start
print("Time to train and save model =",TrainingSavingModel," secs") 


## loading each classifier
print("Loading models...")
start = time.time()
with open('classifier_1.pkl', 'rb') as fid:
    classifier_1 = cPickle.load(fid) 
    
with open('classifier_2.pkl', 'rb') as fid:
    classifier_2 = cPickle.load(fid)

with open('classifier_3.pkl', 'rb') as fid:
    classifier_3 = cPickle.load(fid)

with open('classifier_4.pkl', 'rb') as fid:
    classifier_4 = cPickle.load(fid)

with open('classifier_5.pkl', 'rb') as fid:
    classifier_5 = cPickle.load(fid)

with open('classifier_6.pkl', 'rb') as fid:
    classifier_6 = cPickle.load(fid)

with open('classifier_7.pkl', 'rb') as fid:
    classifier_7 = cPickle.load(fid)

with open('classifier_8.pkl', 'rb') as fid:
    classifier_8 = cPickle.load(fid)
    
end = time.time()
ModelLoadTime = end - start
print("Time to load all models =",ModelLoadTime," secs") 


## recursive function to predict text class using all trained models
def classify_text(text,i=0):
    
    global distinct_classes,class_freq,classifier
    
    ## condition to exit the recursion
    ## after all classifiers are exhausted to classify
    if i == len(distinct_classes):
        return -1
    else:
        ## assign the current class model to 'classifier'
        if ('classifier_'+class_freq['index'][i]) in globals():
            classsifier_assignemnt[int(class_freq['index'][i])]()             
        else: 
            ## load model if not loaded already
            with open('classifier_'+class_freq['index'][i]+'.pkl', 'rb') as fid:
                classifier = cPickle.load(fid) 
                
        ## predict using the current classifier        
        predicted_class = classifier.classify(text)
     
        if predicted_class == '-1': 
            ## if current classifier predicts docuemnt belong to other class
            ## loop again with next classifier to predict
            return classify_text(text,i + 1)
        else: 
            ## return the class label if current classifier predicts document
            ## to belong to current class
            return predicted_class


## defining an object as classifier
classifier = object
## define the function blocks to initialise classifiers
def c_1():
    global classifier
    classifier = classifier_1   

def c_2():
    global classifier
    classifier = classifier_2
    
def c_3():
    global classifier
    classifier = classifier_3
    
def c_4():
    global classifier
    classifier = classifier_4
    
def c_5():
    global classifier
    classifier = classifier_5
    
def c_6():
    global classifier
    classifier = classifier_6
    
def c_7():
    global classifier
    classifier = classifier_7
    
def c_8():
    global classifier
    classifier = classifier_8

## maps the inputs to the function blocks
classsifier_assignemnt = {
    1 : c_1,
    2 : c_2,
    3 : c_3,
    4 : c_4,
    5 : c_5,
    6 : c_6,
    7 : c_7,
    8 : c_8,
}



print("Classifying test data...")

## Predicting on Test Dataset
i=1
test = test.reset_index()
predicted_category =list()

start = time.time()

## iterates through each document in test set
## predicts class and stores prediction in a list
for text in test['Document']:
    text = ' '.join([stem(word) for word in text.split() if word not in stop])
    predicted_category.append(classify_text(text))   
    print("Currently running: ", i)
    i+=1

end = time.time()
elapsed = end - start
AverageClassifyTime = elapsed/(i-1)
print("Average time to classify a document =",AverageClassifyTime," secs")

## adds the list to test dataframe
test['predicted_category'] = pd.Series(predicted_category)


## Confusion Matrix for the actual and predicted value
from sklearn.metrics import confusion_matrix
confusion_mat = confusion_matrix(test['category'].values, test['predicted_category'].values)
print(confusion_mat)


## Sensitivity and Specificity for the actual and predicted value
from sklearn.metrics import classification_report
target_names = ['class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6', 'class 7', 'class 8']
print(classification_report(test['category'].values, test['predicted_category'].values, target_names=target_names))


## Overall Accuracy for the actual and predicted value
from sklearn.metrics import accuracy_score
Accuracy = accuracy_score(test['category'].values, test['predicted_category'].values, normalize=True)
print("Overall Accuracy: ", round(Accuracy*100,2))