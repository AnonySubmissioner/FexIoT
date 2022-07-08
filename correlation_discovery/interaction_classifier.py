import spacy, xlrd, math, os, json
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm
from spacy import displacy
from itertools import chain
from nltk.corpus import wordnet
from dtaidistance import dtw_ndim
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import precision_recall_fscore_support

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier

from joblib import dump, load

def extract_verb(doc):
    verb_list=[]
    for token in doc:
        if token.pos_ == 'VERB':
            verb_list.append(token.text)
    return verb_list

def extract_object(doc):
    wd_list = [nc for nc in doc.noun_chunks]
    object_list=[]
    for span in wd_list:
        str_chunk = str(span).split(" ")
        for ss in str_chunk:
            object_list.append(ss)
    return object_list

def token2vector(doc, word_list):
    vector=[]
    for token in doc:
        for text in word_list:
            if token.text == text:
                vector.append(token.vector)
    return vector

def dtw_distance(series1, series2):
    res = dtw_ndim.distance(series1, series2)
    if math.isinf(res):
        res = 0
    elif res!=0:
        res = 1/res
    return [res]

def word_synonym_hypernym(trigger_word_list, action_word_list):
    result_binary=[0, 0]

    for text in action_word_list:
        synonyms = wordnet.synsets(text)
        lemmas = list(set(chain.from_iterable([word.lemma_names() for word in synonyms])))
        hypernyms = list(set(chain.from_iterable([word.hypernyms() for word in synonyms])))
        hypernyms = [k.name().split(".")[0] for k in hypernyms]

        for wd in trigger_word_list:
            if wd in lemmas:
                result_binary[0] = 1
                break

        for wd in trigger_word_list:
            if wd in hypernyms:
                result_binary[1] = 1
                break
    return result_binary
                
def word_meronym_holonym(trigger_word_list, action_word_list):
    result_binary=[0, 0]
    for text in action_word_list:
        synonyms = wordnet.synsets(text)
        part_lemmas = list(set(chain.from_iterable([word.part_meronyms() for word in synonyms])))
        meronyms_word_list = [k.name().split(".")[0] for k in part_lemmas]

        substance_lemmas = list(set(chain.from_iterable([word.substance_meronyms() for word in synonyms])))
        meronyms_word_list.extend([k.name().split(".")[0] for k in substance_lemmas])

        part_holo = list(set(chain.from_iterable([word.part_holonyms() for word in synonyms])))
        holonyms_word_list = [k.name().split(".")[0] for k in part_holo]

        substance_holo = list(set(chain.from_iterable([word.substance_holonyms() for word in synonyms])))
        holonyms_word_list.extend([k.name().split(".")[0] for k in substance_holo])

        for wd in trigger_word_list:
            if wd in meronyms_word_list:
                result_binary[0]=1
                break

        for wd in trigger_word_list:
            if wd in holonyms_word_list:
                result_binary[0]=1
                break

    return result_binary     

def merge_features(trigger, action):
    nlp = spacy.load('en_core_web_lg')
    feature = []

    trigger_doc = nlp(trigger)
    action_doc = nlp(action)

    embed = trigger_doc.vector + action_doc.vector
    feature.extend(embed.tolist())

    trigger_verb = extract_verb(trigger_doc)
    action_verb = extract_verb(action_doc)

    trigger_object = extract_object(trigger_doc)
    action_object = extract_object(action_doc)

    vector_trigger_verb = token2vector(trigger_doc, trigger_verb)
    vector_action_verb = token2vector(action_doc, action_verb)

    vector_trigger_object = token2vector(trigger_doc, trigger_object)
    vector_action_object = token2vector(action_doc, action_object)

    feature.extend(dtw_distance(vector_trigger_object, vector_action_object))
    feature.extend(dtw_distance(vector_trigger_verb, vector_action_verb))
    feature.extend(word_synonym_hypernym(trigger_verb, action_verb))
    feature.extend(word_synonym_hypernym(trigger_object, action_object))
    feature.extend(word_meronym_holonym(trigger_object, action_object))

    feature = np.array(feature)

    return feature

def read_pair(file_path, sheet_name='examples'):
    book = xlrd.open_workbook(file_path)
    sheet1 = book.sheet_by_name(sheet_name)

    action_list = sheet1.col_values(0)
    trigger_list = sheet1.col_values(1)
    label_list = sheet1.col_values(2)
    
    dataset=[]
    for i in range(len(action_list)):
        dataset.append(np.array([merge_features(trigger_list[i], action_list[i]), label_list[i]], dtype=object))
    return dataset

def classification(dataset, model_names = ["KNN", "MLP", "RandomForest", "GradientBoost"]):
    feature = np.array([dataset[i][0] for i in range(len(dataset))])
    label = np.array([dataset[i][1] for i in range(len(dataset))])

    kf = KFold(n_splits=10, shuffle = True, random_state = 42)

    classifiers = [
        KNeighborsClassifier(3),
        MLPClassifier(hidden_layer_sizes=(512, 400, 300, 256, 128, 64), alpha=0.0001, max_iter=10000000, learning_rate = "adaptive", learning_rate_init = 0.00001,  activation = "relu", random_state =42),
        RandomForestClassifier(n_estimators=1000, max_features="auto", criterion = 'entropy'),
        GradientBoostingClassifier(loss = 'deviance', n_estimators=512, learning_rate=0.01, random_state=42, max_depth = 1000)
        ]

    # iterate over classifiers
    accuracy_dict={"KNN":[], "MLP":[], "RandomForest":[], "GradientBoost":[]}
    precision_dict={"KNN":[], "MLP":[], "RandomForest":[], "GradientBoost":[]}
    recall_dict={"KNN":[], "MLP":[], "RandomForest":[], "GradientBoost":[]}
    fscore_dict={"KNN":[], "MLP":[], "RandomForest":[], "GradientBoost":[]}
    for name, clf in zip(model_names, classifiers):
        print("I am training ", name)

        for train_index, test_index in kf.split(feature):
            X_train, X_test = feature[train_index], feature[test_index]
            y_train, y_test = label[train_index], label[test_index]

            model_file = "./saved_models/"+name+'_interaction_classifier.joblib'
            if os.path.exists(model_file):
                clf = load(model_file)
            else:
                clf.fit(X_train, y_train)
                dump(clf, model_file)
            accuracy = clf.score(X_test, y_test)
            y_pred = clf.predict(X_test)
            results = precision_recall_fscore_support(y_test, y_pred, average='binary',  pos_label=1)
            accuracy_dict[name].append(accuracy) 
            precision_dict[name].append(results[0]) 
            recall_dict[name].append(results[1]) 
            fscore_dict[name].append(results[2])
        
    # create json object from dictionary

    for measure in ["accuracy", "recall", "precision", "fscore"]:
        name = locals()[measure+'_dict'] # Convert String to Variable Name
        f = open("results/"+measure+".json","w")
        f.write(json.dumps(name))
        f.close()

if __name__=="__main__":
    initial = False
    train2data = True
    apply2data = False, 
    random_state=42,

    if train2data == True:
        if initial == True:
            dataset = read_pair('./trigger-action.xls', 'examples')
            dataset = np.array(dataset, dtype=object)
            np.save('./feature_label_dataset.npy', dataset)
            print("I generated features")
        else:
            dataset = np.load('./feature_label_dataset.npy',  allow_pickle = True)
            print("I loaded features")

        classification(dataset)
