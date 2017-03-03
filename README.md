# Text classification with Python
A text classifier for multi-class, highly imbalanced dataset.

##Dataset
For dataset I used a stack of documents, some of them have already been processed. You can find the subset of data in repo as **'text_data.txt'**, with only 1000 records.

In input file, the first line will contain T the number of documents. T lines will follow each containing a category(number 1-8) and series of space seperated words which represents the processed document.

###Sample Input
```
3 
1 This is a document 
2 this is another document 
5 documents are seperated by newlines
```
The sample input file above have 3 documents belonging to category 1,2 and 5 respectively.

##Requirements
  - python 2.7
  - python modules:
    - scikit-learn
    - textblob
    - pandas
    - nltk
    - stemming
    - _pickle
    - math
    - io
    - os
    - time
    
    
##The code
The code is pretty straight forward and well documented.
 
##Running the code
```
multiclass_classifier.py
```

##Algorithm
I have used Naive Bayes classifier for classification.
TFIDF representation is used in preprocessing where words lower than 0.0 tfidf values are removed from dataset. This helped to reduce the corpus size and saved a lot in time in training and a better performance as well.

The other issue I had that the dataset is highly imbalanced with multiple classes. Below is the distribution of dataset along with thier classes:
```
class Count    %
1     548      49.954421
2     340      30.993619
3     49       4.466727
6     49       4.466727
8     45       4.102097
7     33       3.008204
4     20       1.823154
5     13       1.185050
```
FOr dealing with such dataset, I have created a recursive process, wherein the classifier is trained for binary class classification : class 1 vs all other. Next, class1 is removed from training set and classifier is trained on class2 vs all other and so on. This is iterated on the basis of class density rather than class name. This way the classifier is trained on a fairly balanced dataset with just two classes. 

##Performance
The algorithm using naive bayes performs fairly well. Using tfidf, the performance improved a bit.
The algorithm is giving a overall accuracy in mid 80's. With precision and recall also in range of 80's for each class. 
For this highly imbalanced dataset, these measures looks failry good.

##Next what?
Next, I will be trying different combinations of feature-vectors, classifier and train-test split strategy.
  - Feature-vectors: BOW, TF, TFIDF, 
  - Classifier: NB, SVM, kNN
  - Split Strategy: 70-30, 80-20, Kfold

##Feedback
Comments, bug reports, and ideas are more than welcome.
