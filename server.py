from flask import Flask, g, render_template, request, abort, send_from_directory
import numpy as np
from sklearn import metrics
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn import svm
import random
import re
import os
from sklearn.externals import joblib
import pandas as pd

def classifaction_report_csv(report, label):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        m = re.search(' *(\w{4,10}) *(\d{1}\.\d{2}) *(\d{1}\.\d{2}) *(\d{1}\.\d{2}) *(\d{1,})', line)
        row = {}
        if m is None:
            return
        row['class'] = m.group(1).replace("_", " ")
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

#### Captura de Features Conjunto de Treino ####
#Duvida: porque devo encaixar esses tokenizers no conjunto de treino?
X_train_counts = count_vect.fit_transform(docs_train)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

#### Captura de Features Conjunto de Validacao ####
X_validation_counts = count_vect.transform(docs_validation)
X_validation_tfidf = tfidf_transformer.transform(X_validation_counts)

#### Captura de Features Conjunto de Teste ####
X_test_counts = count_vect.transform(docs_test)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

#### Treinar Modelo ####
print "Calibrando Naive Bayers..."
# Alternativa 1: Naive bayes
#Encontrar melhor valor de alpha
alpha_nb = 0
best_accuracy_nb = 0
for x in np.arange(0.1, 1.0, 0.3):
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

# Alternativa 2: SGD
# #
# Existem muitos parametros no sgd do sklearn
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
print "Calibrando SGD..."

# Encontrando o melhor valor de loss
best_accuracy_sgd = 0
loss = ["hinge", "log", "modified_huber", "squared_hinge", "perceptron", "squared_loss", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"]
for l in loss:
    sgd = SGDClassifier(loss=l)
    clf_sgd = sgd.fit(X_validation_tfidf, y_validation)

    # Se foi a maior acuracia ate agora, salva como melhor loss
    predict_validation_sgd = clf_sgd.predict(X_validation_tfidf)
    accuracy_sgd = np.mean(predict_validation_sgd == y_validation)
    
    if accuracy_sgd > best_accuracy_sgd:  
        loss_sgd = l
        best_accuracy_sgd = accuracy_sgd

#Encontrar melhor valor de alpha
alpha_sgd = 0
best_accuracy_sgd = 0
for x in np.arange(0.0001, 1.0, 0.3):
    sgd = SGDClassifier(alpha=x)
    clf_sgd = sgd.fit(X_validation_tfidf, y_validation)

    # Se foi a maior acuracia ate agora, salva como melhor alpha
    predict_validation_sgd = clf_sgd.predict(X_validation_tfidf)
    accuracy_sgd = np.mean(predict_validation_sgd == y_validation)
    
    if accuracy_sgd > best_accuracy_sgd:  
        alpha_sgd = x
        best_accuracy_sgd = accuracy_sgd


# Encontrando o melhor valor de penalty
best_accuracy_sgd = 0
penalty = ["none", "l2", "l1", "elasticnet"]
for p in penalty:
    sgd = SGDClassifier(penalty=p)
    clf_sgd = sgd.fit(X_validation_tfidf, y_validation)
    # Se foi a maior acuracia ate agora, salva como melhor penalty
    predict_validation_sgd = clf_sgd.predict(X_validation_tfidf)
    accuracy_sgd = np.mean(predict_validation_sgd == y_validation)
    
    if accuracy_sgd > best_accuracy_sgd:  
        penalty_sgd = p
        best_accuracy_sgd = accuracy_sgd

# Encontrando o melhor valor de learning_rate
best_accuracy_sgd = 0
learning_rate = ["constant","optimal","invscaling"]
for lr in learning_rate:
    eta0_sgd = 1 #chutando um valor de eta0
    sgd = SGDClassifier(learning_rate=lr, eta0=eta0_sgd)
    clf_sgd = sgd.fit(X_validation_tfidf, y_validation)
    # Se foi a maior acuracia ate agora, salva
    predict_validation_sgd = clf_sgd.predict(X_validation_tfidf)
    accuracy_sgd = np.mean(predict_validation_sgd == y_validation)

    if accuracy_sgd > best_accuracy_sgd:  
        learning_rate_sgd = lr
        best_accuracy_sgd = accuracy_sgd
if learning_rate_sgd is not "optimal":
    #Necessario aprender melhor valor de eta0
    best_accuracy_sgd = 0
    for x in np.arange(0.01, 1.0, 0.3):
        sgd = SGDClassifier(learning_rate=learning_rate_sgd,
                            eta0=x)
        clf_sgd = sgd.fit(X_validation_tfidf, y_validation)

        #se foi a maior acuracia ate agora, salva
        predict_validation_sgd = clf_sgd.predict(X_validation_tfidf)
        accuracy_sgd = np.mean(predict_validation_sgd == y_validation)
        
        if accuracy_sgd > best_accuracy_sgd:  
            eta0_sgd = x
            best_accuracy_sgd = accuracy_sgd

# Encontrando o melhor valor de tol (Criterio de parada)
best_accuracy_sgd = 0
for x in np.arange(0.001, 2.1, 0.01):
    sgd = SGDClassifier(tol=x)
    clf_sgd = sgd.fit(X_validation_tfidf, y_validation)
    #Se foi a maior acuracia ate agora, salva 
    predict_validation_sgd = clf_sgd.predict(X_validation_tfidf)
    accuracy_sgd = np.mean(predict_validation_sgd == y_validation)
    if accuracy_sgd > best_accuracy_sgd:
        tol_sgd = x
        best_accuracy_sgd = accuracy_sgd

# Encontrar o melhor valor de max interacoes nao faz muita diferenca
# best_accuracy_sgd = 0
# for x in range(5, 100, 1):
#     sgd = SGDClassifier(loss=loss_sgd, penalty=penalty_sgd,
#                         alpha=alpha_sgd, random_state=42,
#                         max_iter=x,learning_rate=learning_rate_sgd,
#                         eta0=eta0_sgd,tol=tol_sgd)
#     clf_sgd = sgd.fit(X_validation_tfidf, y_validation)
#     #Se foi a maior acuracia ate agora, salva 
#     predict_validation_sgd = clf_sgd.predict(X_validation_tfidf)
#     accuracy_sgd = np.mean(predict_validation_sgd == y_validation)
#     if accuracy_sgd >= best_accuracy_sgd:
#         max_iter_sgd = x
#         best_accuracy_sgd = accuracy_sgd

sgd = SGDClassifier(loss=loss_sgd,penalty=penalty_sgd,
                    alpha=alpha_sgd, random_state=42,
                    max_iter=5, tol=tol_sgd,
                    learning_rate=learning_rate_sgd, eta0=eta0_sgd)

clf_sgd = sgd.fit(X_train_tfidf, y_train)

#### Avaliando algoritmo ####
predict_test_sgd = clf_sgd.predict(X_test_tfidf)
accuracy_sgd = np.mean(predict_test_sgd == y_test)

report_sgd = metrics.classification_report(y_test, predict_test_sgd, target_names=category)
classifaction_report_csv(report_sgd,"sgd")
#print(metrics.confusion_matrix(y_test, predict_test_sgd))

# Alternativa 3: SVM
# LinearSVC e outra implementacao de Support Vector Classification 
# para o caso de kernel linear.
print "Calibrando SVM..."
# Encontrando o melhor valor de penalty
best_accuracy_svm = 0
penalty = ["l2", "l1"]
for p in penalty:
    svm_lin = svm.LinearSVC(penalty=p, dual=False)
    clf_svm = svm_lin.fit(X_validation_tfidf, y_validation)
    # Se foi a maior acuracia ate agora, salva como melhor penalty
    predict_validation_svm = clf_svm.predict(X_validation_tfidf)
    accuracy_svm = np.mean(predict_validation_svm == y_validation)
    
    if accuracy_svm > best_accuracy_svm:  
        penalty_svm = p
        best_accuracy_svm = accuracy_svm

best_accuracy_svm = 0
loss = ["hinge", "squared_hinge"]
for l in loss:
    svm_lin = svm.LinearSVC(loss=l)
    clf_svm = svm_lin.fit(X_validation_tfidf, y_validation)
    # Se foi a maior acuracia ate agora, salva como melhor loss
    predict_validation_svm = clf_svm.predict(X_validation_tfidf)
    accuracy_svm = np.mean(predict_validation_svm == y_validation)
    
    if accuracy_svm > best_accuracy_svm:
        loss_svm = l
        best_accuracy_svm = accuracy_svm

best_accuracy_svm = 0
multi_class = ["ovr", "crammer_singer"]
for mc in multi_class:
    svm_lin = svm.LinearSVC(multi_class=mc)
    clf_svm = svm_lin.fit(X_validation_tfidf, y_validation)
    # Se foi a maior acuracia ate agora, salva como melhor multi class
    predict_validation_svm = clf_svm.predict(X_validation_tfidf)
    accuracy_svm = np.mean(predict_validation_svm == y_validation)
    
    if accuracy_svm > best_accuracy_svm:
        mc_svm = mc
        best_accuracy_svm = accuracy_svm

best_accuracy_svm = 0
dual = [True, False]
for d in dual:
    svm_lin = svm.LinearSVC(dual=d)
    clf_svm = svm_lin.fit(X_validation_tfidf, y_validation)
    # Se foi a maior acuracia ate agora, salva
    predict_validation_svm = clf_svm.predict(X_validation_tfidf)
    accuracy_svm = np.mean(predict_validation_svm == y_validation)
    
    if accuracy_svm > best_accuracy_svm:
        dual_svm = d
        best_accuracy_svm = accuracy_svm

clf_svm = svm.LinearSVC(dual=dual_svm, loss=loss_svm,
     multi_class=mc_svm, penalty=penalty_svm, verbose=0, max_iter=3000)
clf_svm.fit(X_train_tfidf, y_train) 

predict_test_svm = clf_svm.predict(X_test_tfidf)
accuracy_svm = np.mean(predict_test_svm == y_test)

report_svm = metrics.classification_report(y_test, predict_test_svm, target_names=category)
classifaction_report_csv(report_svm,"svm")

#### Salvando modelo ####
joblib.dump(clf_nb, 'model_nb.pkl') 
joblib.dump(clf_sgd, 'model_sgd.pkl') 
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
    sgd = {};
    svm = {};

    X_new_counts = count_vect.transform(q)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    #1
    nb["predict"] = category[clf_nb.predict(X_new_tfidf)].replace("_", " ")
    nb["accuracy"] = accuracy_nb
    #2
    sgd["predict"] = category[clf_sgd.predict(X_new_tfidf)].replace("_", " ")
    sgd["accuracy"] = accuracy_sgd
    #3
    svm["predict"] = category[clf_svm.predict(X_new_tfidf)].replace("_", " ")
    svm["accuracy"] = accuracy_svm

    return render_template('results.html', nb=nb, sgd=sgd, svm=svm)

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
