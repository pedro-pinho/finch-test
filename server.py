from flask import Flask, g, render_template, request, abort, send_from_directory
import numpy as np
from sklearn import metrics
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
import random
import re
import os
from sklearn.externals import joblib
import pandas as pd

def classifaction_report_csv(report, label):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        m = re.search(' *(\w{4,10}) *(\d{1}\.\d{2}) *(\d{1}\.\d{2}) *(\d{1}\.\d{2}) *(\d{3})', line)
        row = {}
        row['class'] = m.group(1)
        row['precision'] = float(m.group(2))
        row['recall'] = float(m.group(3))
        row['f1_score'] = float(m.group(4))
        row['support'] = float(m.group(5))
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv(label+'.csv', index = False)

#### Pre-processamento de dados ####
#
category = ['bossa_nova','funk','gospel','sertanejo']

for c in category:
    i=0
    for filename in os.listdir(os.path.join(os.path.dirname(__file__),'data',c)):
        i+=1
    ## Se nunca passou pelo pre-processamento
    if(i<=4):
        for filename in os.listdir(os.path.join(os.path.dirname(__file__),'data',c)):
            #abre cada arquivo, enorme
            file=open(os.path.join(os.path.dirname(__file__),'data',c,filename),'r')
            fileContent=file.read()
            #separa usando uma regex
            myregex = re.compile('\"\n^(" \n)',re.M)
            lyricList = myregex.split(fileContent)
            #salva arquivos picados
            j=0
            for lyric in lyricList:
                lyric = lyric.replace("lyric", "", 3)
                lyric = lyric.replace('"', '', 3)
                if len(lyric)>2:
                    j+=1
                    f=open("data/"+c+"/"+str(j)+".txt","w+")
                    f.write(lyric)
                    f.close()
            os.remove(os.path.join(os.path.dirname(__file__),'data',c,filename))

#### Carregando arquivos ####
# mydata = np.genfromtxt(filename, delimiter=",")
dataset = load_files('data', encoding='ISO-8859-1', load_content=True, categories=category)

# 80% treino
test_size = 0.2
docs_train, docs_to_split, y_train, y_to_split = train_test_split(
    dataset.data, dataset.target, test_size = test_size, random_state=1)

#10% teste, 10% validacao
validation_size = 0.5
docs_test, docs_validation, y_test, y_validation = train_test_split(
    docs_to_split, y_to_split, test_size = validation_size, random_state=1)

# Tokenizer
count_vect = CountVectorizer()

# Abordagem tf-idf
tfidf_transformer = TfidfTransformer()

#### Captura de Features Conjunto de Validacao ####
X_validation_counts = count_vect.transform(docs_validation)
X_validation_tfidf = tfidf_transformer.transform(X_validation_counts)

#### Captura de Features Conjunto de Treino ####
X_train_counts = count_vect.fit_transform(docs_train)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

#### Captura de Features Conjunto de Teste ####
X_test_counts = count_vect.transform(docs_test)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

#### Treinar Modelo ####

# Alternativa 1: Naive bayes
#Encontrar melhor valor de alpha
alpha_nb = 0
best_accuracy_nb = 0
for x in np.arange(0.0, 1.0, 0.3):
    clf_nb = MultinomialNB(alpha=x).fit(X_validation_tfidf, y_validation)
    
    predict_validation_nb = clf_nb.predict(X_validation_tfidf)
    accuracy_nb = np.mean(predict_validation_nb == y_validation)

    #se foi a maior acuracia ate agora, salva como melhor alpha
    if accuracy_nb > best_accuracy_nb:  
        alpha_nb = x
        best_accuracy_nb = accuracy_nb

#Com o alpha encontrado
clf_nb = MultinomialNB(alpha=alpha_nb).fit(X_train_tfidf, y_train)

#### Avaliando algoritmo ####
predict_test_nb = clf_nb.predict(X_test_tfidf)
accuracy_nb = np.mean(predict_test_nb == y_test)

report_nb=metrics.classification_report(y_test, predict_test_nb, target_names=category) 
classifaction_report_csv(report_nb,"nb")
#print(metrics.confusion_matrix(y_test, predict_test_nb))

# Alternativa 2: SVM
# #
# Existem muitos parametros no svm do sklearn
# loss=hinge,
# penalty=l2
# alpha=0.0001
# l1_ratio=0.15
# fit_intercept=True
# max_iter=None
# tol=None
# shuffle=True
# verbose=0
# epsilon=0.1,
# n_jobs=1,
# random_state=None,
# learning_rate=optimal,
# eta0=0.0,
# power_t=0.5,
# class_weight=None,
# warm_start=False,
# average=False,
# n_iter=None
# # 
#Encontrar melhor valor de alpha
alpha_svm = 0
best_accuracy_svm = 0
for x in np.arange(0.0001, 1.0, 0.3):
    svm = SGDClassifier(loss='hinge', penalty='l2', alpha=x, random_state=42, max_iter=5, tol=None)
    clf_svm = svm.fit(X_validation_tfidf, y_validation)

    #se foi a maior acuracia ate agora, salva como melhor alpha
    predict_validation_svm = clf_svm.predict(X_validation_tfidf)
    accuracy_svm = np.mean(predict_validation_svm == y_validation)
    
    if accuracy_svm > best_accuracy_svm:  
        alpha_svm = x
        best_accuracy_svm = accuracy_svm


svm = SGDClassifier(loss='hinge', penalty='l2', alpha=alpha_svm, random_state=42, max_iter=5, tol=None)
clf_svm = svm.fit(X_train_tfidf, y_train)

#### Avaliando algoritmo ####
# print "Metricas Support Vector Machine(SVM)"
predict_test_svm = clf_svm.predict(X_test_tfidf)
accuracy_svm = np.mean(predict_test_svm == y_test)

report_svm = metrics.classification_report(y_test, predict_test_svm, target_names=category)
classifaction_report_csv(report_svm,"svm")
# print report_svm

#print(metrics.confusion_matrix(y_test, predict_test_svm))

#### Salvando modelo ####
joblib.dump(clf_nb, 'model_nb.pkl') 
joblib.dump(clf_svm, 'model_svm.pkl') 

app = Flask(__name__)

@app.route('/')
def index():
    return render_template(
        'index.html'
    )


@app.route('/predict', methods=['POST'])
def predict():
    q = [request.form['q']] or ['']
    nb = {};
    svm = {};

    X_new_counts = count_vect.transform(q)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    #1
    nb["predict"] = category[clf_nb.predict(X_new_tfidf)]
    nb["accuracy"] = accuracy_nb
    #2
    svm["predict"] = category[clf_svm.predict(X_new_tfidf)]
    svm["accuracy"] = accuracy_svm

    return render_template('results.html', nb=nb, svm=svm)

@app.route('/js/<path:path>')
def js(path):
    return send_from_directory('csv-to-html-table/js', path)

@app.route('/csv/<path:path>')
def csv(path):
    return send_from_directory('', path)

@app.route('/css/<path:path>')
def css(path):
    return send_from_directory('csv-to-html-table/css', path)

@app.route('/fonts/<path:path>')
def fonts(path):
    return send_from_directory('csv-to-html-table/fonts', path)

if __name__ == '__main__':
    app.run(port=5000,host='0.0.0.0',debug=True)
