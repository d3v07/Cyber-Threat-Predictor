from django.db.models import Count
from django.db.models import Q
from django.shortcuts import render, redirect, get_object_or_404
import datetime
import openpyxl

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
# Create your views here.
from Remote_User.models import ClientRegister_Model,detect_cyber_threat,detection_ratio,detection_accuracy

def login(request):


    if request.method == "POST" and 'submit1' in request.POST:

        username = request.POST.get('username')
        password = request.POST.get('password')
        try:
            enter = ClientRegister_Model.objects.get(username=username,password=password)
            request.session["userid"] = enter.id

            return redirect('ViewYourProfile')
        except:
            pass

    return render(request,'RUser/login.html')

def Add_DataSet_Details(request):

    return render(request, 'RUser/Add_DataSet_Details.html', {"excel_data": ''})


def Register1(request):

    if request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        phoneno = request.POST.get('phoneno')
        country = request.POST.get('country')
        state = request.POST.get('state')
        city = request.POST.get('city')
        ClientRegister_Model.objects.create(username=username, email=email, password=password, phoneno=phoneno,
                                            country=country, state=state, city=city)

        return render(request, 'RUser/Register1.html')
    else:
        return render(request,'RUser/Register1.html')

def ViewYourProfile(request):
    userid = request.session['userid']
    obj = ClientRegister_Model.objects.get(id= userid)
    return render(request,'RUser/ViewYourProfile.html',{'object':obj})


def Predict_Cyber_Threat_Type(request):
    if request.method == "POST":

        if request.method == "POST":

            pid= request.POST.get('pid')
            ptime= request.POST.get('ptime')
            src_ip_address= request.POST.get('src_ip_address')
            dst_ip_address= request.POST.get('dst_ip_address')
            frame_protos= request.POST.get('frame_protos')
            src_port= request.POST.get('src_port')
            dst_port= request.POST.get('dst_port')
            bytes_trans= request.POST.get('bytes_trans')
            protocol= request.POST.get('protocol')
            Date1= request.POST.get('Date1')


        dataset = pd.read_csv("IIoT_Network_Datasets.csv", encoding='latin-1')

        def apply_results(label):
            if (label == 0):
                return 0  # Packet Drop
            elif (label == 1):
                return 1  # Packet hijacking

        dataset['Results'] = dataset['attack'].apply(apply_results)

        cv = CountVectorizer()

        x = dataset['pid'].apply(str)
        y = dataset['Results']

        cv = CountVectorizer()

        print(x)
        print("Y")
        print(y)

        x = cv.fit_transform(x)
        models = []
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
        X_train.shape, X_test.shape, y_train.shape

        print("Naive Bayes")

        from sklearn.naive_bayes import MultinomialNB
        NB = MultinomialNB()
        NB.fit(X_train, y_train)
        predict_nb = NB.predict(X_test)
        naivebayes = accuracy_score(y_test, predict_nb) * 100
        print(naivebayes)
        print(confusion_matrix(y_test, predict_nb))
        print(classification_report(y_test, predict_nb))
        models.append(('naive_bayes', NB))

        # SVM Model
        print("SVM")
        from sklearn import svm
        lin_clf = svm.LinearSVC()
        lin_clf.fit(X_train, y_train)
        predict_svm = lin_clf.predict(X_test)
        svm_acc = accuracy_score(y_test, predict_svm) * 100
        print(svm_acc)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, predict_svm))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, predict_svm))
        models.append(('svm', lin_clf))

        print("Logistic Regression")

        from sklearn.linear_model import LogisticRegression
        reg = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, y_pred) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, y_pred))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, y_pred))
        models.append(('logistic', reg))

        classifier = VotingClassifier(models)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        pid1 = [pid]
        vector1 = cv.transform(pid1).toarray()
        predict_text = classifier.predict(vector1)

        pred = str(predict_text).replace("[", "")
        pred1 = pred.replace("]", "")

        prediction = int(pred1)

        if prediction == 0:
            val = 'Packet Drop'
        elif prediction == 1:
            val = 'Packet Hijacking'


        print(val)
        print(pred1)

        detect_cyber_threat.objects.create(
        pid=pid,
        ptime=ptime,
        src_ip_address=src_ip_address,
        dst_ip_address=dst_ip_address,
        frame_protos=frame_protos,
        src_port=src_port,
        dst_port=dst_port,
        bytes_trans=bytes_trans,
        protocol=protocol,
        Date1=Date1,
        Prediction=val)

        return render(request, 'RUser/Predict_Cyber_Threat_Type.html',{'objs': val})
    return render(request, 'RUser/Predict_Cyber_Threat_Type.html')



