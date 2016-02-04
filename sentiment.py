import sys
import collections
import sklearn.naive_bayes
import sklearn.linear_model
import nltk
import random
random.seed(0)
from gensim.models.doc2vec import LabeledSentence, Doc2Vec
#nltk.download("stopwords")          # Download the stop words from nltk


# User input path to the train-pos.txt, train-neg.txt, test-pos.txt, and test-neg.txt datasets
if len(sys.argv) != 3:
    print "python sentiment.py <path_to_data> <0|1>"
    print "0 = NLP, 1 = Doc2Vec"
    exit(1)
path_to_data = sys.argv[1]
method = int(sys.argv[2])



def main():
    train_pos, train_neg, test_pos, test_neg = load_data(path_to_data)
    
    if method == 0:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_NLP(train_pos_vec, train_neg_vec)
    if method == 1:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_DOC(train_pos_vec, train_neg_vec)
    print "Naive Bayes"
    print "-----------"
    evaluate_model(nb_model, test_pos_vec, test_neg_vec, True)
    print ""
    print "Logistic Regression"
    print "-------------------"
    evaluate_model(lr_model, test_pos_vec, test_neg_vec, True)



def load_data(path_to_dir):
    """
    Loads the train and test set into four different lists.
    """
    train_pos = []
    train_neg = []
    test_pos = []
    test_neg = []
    with open(path_to_dir+"train-pos.txt", "r") as f:
        for i,line in enumerate(f):
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_pos.append(words)
    with open(path_to_dir+"train-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_neg.append(words)
    with open(path_to_dir+"test-pos.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_pos.append(words)
    with open(path_to_dir+"test-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_neg.append(words)

    return train_pos, train_neg, test_pos, test_neg



def feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # English stopwords from nltk
    stopwords = set(nltk.corpus.stopwords.words('english'))

    # Determine a list of words that will be used as features.
    # This list should have the following properties:
    #   (1) Contains no stop words
    #   (2) Is in at least 1% of the positive texts or 1% of the negative texts
    #   (3) Is in at least twice as many postive texts as negative texts, or vice-versa.
    # YOUR CODE HERE
    pos_counter = collections.Counter()
    pos_len = len(train_pos)
    for one in train_pos:
      for wrd in set(one):
        if wrd in stopwords:
          continue
        pos_counter[wrd] = pos_counter.get(wrd, 0) + 1

    neg_counter = collections.Counter()
    neg_len = len(train_neg)
    for one in train_neg:
      for wrd in set(one):
        if wrd in stopwords:
          continue
        neg_counter[wrd] = neg_counter.get(wrd, 0) + 1

    counter = 0
    feature_map = collections.OrderedDict()
    for key, pos_val in pos_counter.items():
      neg_val = neg_counter.get(key, 0)
      if pos_val > pos_len*0.01 and pos_val >= 2*neg_val:
        feature_map[key] = counter
        counter+=1

    for key, neg_val in neg_counter.items():
      pos_val = pos_counter.get(key, 0)
      if neg_val > neg_len*0.01 and neg_val >= 2*pos_val:
        feature_map[key] = counter
        counter+=1

    feature_size = len(feature_map.keys())

    # Using the above words as features, construct binary vectors for each text in the training and test set.
    # These should be python lists containing 0 and 1 integers.
    # YOUR CODE HERE
    train_pos_vec = []
    for one in train_pos:
      pos_vector = [0]*feature_size
      for wrd in set(one):
        if wrd in stopwords:
          continue
        pos_index = feature_map.get(wrd, -1)
        if pos_index >= 0:
          pos_vector[pos_index] = 1
      train_pos_vec.append(pos_vector)

    train_neg_vec = []
    for one in train_neg:
      neg_vector = [0]*feature_size
      for wrd in set(one):
        if wrd in stopwords:
          continue
        neg_index = feature_map.get(wrd, -1)
        if neg_index >= 0:
          neg_vector[neg_index] = 1
      train_neg_vec.append(neg_vector)

    test_pos_vec = []
    for one in test_pos:
      pos_vector = [0]*feature_size
      for wrd in set(one):
        if wrd in stopwords:
          continue
        pos_index = feature_map.get(wrd, -1)
        if pos_index >= 0:
          pos_vector[pos_index] = 1
      test_pos_vec.append(pos_vector)

    test_neg_vec = []
    for one in test_neg:
      neg_vector = [0]*feature_size
      for wrd in set(one):
        if wrd in stopwords:
          continue
        neg_index = feature_map.get(wrd, -1)
        if neg_index >= 0:
          neg_vector[neg_index] = 1
      test_neg_vec.append(neg_vector)

    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



def feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # Doc2Vec requires LabeledSentence objects as input.
    # Turn the datasets from lists of words to lists of LabeledSentence objects.
    labeled_train_pos=[]
    for index, line in enumerate(train_pos):
      labeled_train_pos.append(LabeledSentence(words=line, tags=['TRAIN_POS_%d'%index]))

    labeled_train_neg=[]
    for index, line in enumerate(train_neg):
      labeled_train_neg.append(LabeledSentence(words=line, tags=['TRAIN_NEG_%d'%index]))

    labeled_test_pos=[]
    for index, line in enumerate(test_pos):
      labeled_test_pos.append(LabeledSentence(words=line, tags=['TEST_POS_%d'%index]))

    labeled_test_neg=[]
    for index, line in enumerate(test_neg):
      labeled_test_neg.append(LabeledSentence(words=line, tags=['TEST_NEG_%d'%index]))

    # Initialize model
    model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=4)
    sentences = labeled_train_pos + labeled_train_neg + labeled_test_pos + labeled_test_neg
    model.build_vocab(sentences)

    # Train the model
    # This may take a bit to run 
    for i in range(5):
        print "Training iteration %d" % i
        random.shuffle(sentences)
        model.train(sentences)

    # Use the docvecs function to extract the feature vectors for the training and test data
    train_pos_vec = []
    for index in range(len(train_pos)):
      train_pos_vec.append(model.docvecs['TRAIN_POS_%d'%index])
    train_neg_vec = []
    for index in range(len(train_neg)):
      train_neg_vec.append(model.docvecs['TRAIN_NEG_%d'%index])
    test_pos_vec = []
    for index in range(len(test_pos)):
      test_pos_vec.append(model.docvecs['TEST_POS_%d'%index])
    test_neg_vec = []
    for index in range(len(test_neg)):
      test_neg_vec.append(model.docvecs['TEST_NEG_%d'%index])
    
    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



def build_models_NLP(train_pos_vec, train_neg_vec):
    """
    Returns a BernoulliNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)
    X = train_pos_vec + train_neg_vec
    # Use sklearn's BernoulliNB and LogisticRegression functions to fit two models to the training data.
    # For BernoulliNB, use alpha=1.0 and binarize=None
    nb_model = sklearn.naive_bayes.BernoulliNB(alpha=1.0, binarize=None)
    nb_model.fit(X, Y)

    # For LogisticRegression, pass no parameters
    lr_model = sklearn.linear_model.LogisticRegression()
    lr_model.fit(X, Y)
    return nb_model, lr_model



def build_models_DOC(train_pos_vec, train_neg_vec):
    """
    Returns a GaussianNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)
    X = train_pos_vec + train_neg_vec
    # Use sklearn's GaussianNB and LogisticRegression functions to fit two models to the training data.
    nb_model = sklearn.naive_bayes.GaussianNB()
    nb_model.fit(X, Y)

    # For LogisticRegression, pass no parameters
    lr_model = sklearn.linear_model.LogisticRegression()
    lr_model.fit(X, Y)
    return nb_model, lr_model



def evaluate_model(model, test_pos_vec, test_neg_vec, print_confusion=False):
    """
    Prints the confusion matrix and accuracy of the model.
    """
    # Use the predict function and calculate the true/false positives and true/false negative.
    tp, tn, fp, fn = 0, 0, 0, 0

    pos_res = model.predict(test_pos_vec)
    for one in pos_res:
      if one == 'pos':
        tp += 1
      else:
        fn += 1

    neg_res = model.predict(test_neg_vec)
    for one in neg_res:
      if one == 'neg':
        tn += 1
      else:
        fp += 1

    accuracy = float(tp + tn) / float(tp + fn + fp + tn)
    if print_confusion:
        print "predicted:\tpos\tneg"
        print "actual:"
        print "pos\t\t%d\t%d" % (tp, fn)
        print "neg\t\t%d\t%d" % (fp, tn)
    print "accuracy: %f" % accuracy



if __name__ == "__main__":
    main()
