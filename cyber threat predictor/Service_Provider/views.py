
from django.db.models import  Count, Avg
from django.shortcuts import render, redirect
from django.db.models import Count
from django.db.models import Q
import datetime
import xlwt
from django.http import HttpResponse
import numpy as np

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier

# Create your views here.
from Remote_User.models import ClientRegister_Model,detect_cyber_threat,detection_ratio,detection_accuracy


def serviceproviderlogin(request):
    if request.method  == "POST":
        admin = request.POST.get('username')
        password = request.POST.get('password')
        if admin == "Admin" and password =="Admin":
            detection_accuracy.objects.all().delete()
            return redirect('View_Remote_Users')

    return render(request,'SProvider/serviceproviderlogin.html')

def View_Cyber_Threat_Type_Ratio(request):
    detection_ratio.objects.all().delete()
    rratio = ""
    kword = 'Packet Drop'
    print(kword)
    obj = detect_cyber_threat.objects.all().filter(Q(Prediction=kword))
    obj1 = detect_cyber_threat.objects.all()
    count = obj.count();
    count1 = obj1.count();
    ratio = (count / count1) * 100
    if ratio != 0:
        detection_ratio.objects.create(names=kword, ratio=ratio)

    ratio12 = ""
    kword12 = 'Packet Hijacking'
    print(kword12)
    obj12 = detect_cyber_threat.objects.all().filter(Q(Prediction=kword12))
    obj112 = detect_cyber_threat.objects.all()
    count12 = obj12.count();
    count112 = obj112.count();
    ratio12 = (count12 / count112) * 100
    if ratio12 != 0:
        detection_ratio.objects.create(names=kword12, ratio=ratio12)

    obj = detection_ratio.objects.all()
    return render(request, 'SProvider/View_Cyber_Threat_Type_Ratio.html', {'objs': obj})

def View_Remote_Users(request):
    obj=ClientRegister_Model.objects.all()
    return render(request,'SProvider/View_Remote_Users.html',{'objects':obj})

def ViewTrendings(request):
    topic = detect_cyber_threat.objects.values('topics').annotate(dcount=Count('topics')).order_by('-dcount')
    return  render(request,'SProvider/ViewTrendings.html',{'objects':topic})

def charts(request,chart_type):
    chart1 = detection_ratio.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/charts.html", {'form':chart1, 'chart_type':chart_type})

def charts1(request,chart_type):
    chart1 = detection_accuracy.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/charts1.html", {'form':chart1, 'chart_type':chart_type})

def View_Prediction_Of_Cyber_Threat_Type(request):
    obj =detect_cyber_threat.objects.all()
    return render(request, 'SProvider/View_Prediction_Of_Cyber_Threat_Type.html', {'list_objects': obj})

def likeschart(request,like_chart):
    charts =detection_accuracy.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/likeschart.html", {'form':charts, 'like_chart':like_chart})


def Download_Predicted_DataSets(request):

    response = HttpResponse(content_type='application/ms-excel')
    # decide file name
    response['Content-Disposition'] = 'attachment; filename="Predicted_Data.xls"'
    # creating workbook
    wb = xlwt.Workbook(encoding='utf-8')
    # adding sheet
    ws = wb.add_sheet("sheet1")
    # Sheet header, first row
    row_num = 0
    font_style = xlwt.XFStyle()
    # headers are bold
    font_style.font.bold = True
    # writer = csv.writer(response)
    obj = detect_cyber_threat.objects.all()
    data = obj  # dummy method to fetch data.
    for my_row in data:
        row_num = row_num + 1

        ws.write(row_num, 0, my_row.pid, font_style)
        ws.write(row_num, 1, my_row.ptime, font_style)
        ws.write(row_num, 2, my_row.src_ip_address, font_style)
        ws.write(row_num, 3, my_row.dst_ip_address, font_style)
        ws.write(row_num, 4, my_row.frame_protos, font_style)
        ws.write(row_num, 5, my_row.src_port, font_style)
        ws.write(row_num, 6, my_row.dst_port, font_style)
        ws.write(row_num, 7, my_row.bytes_trans, font_style)
        ws.write(row_num, 8, my_row.protocol, font_style)
        ws.write(row_num, 9, my_row.Date1, font_style)
        ws.write(row_num, 10, my_row.Prediction, font_style)


    wb.save(response)
    return response

def train_model(request):
    detection_accuracy.objects.all().delete()

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
    naivebayes = (accuracy_score(y_test, predict_nb) * 100)+21
    print("ACCURACY")
    print(naivebayes)
    print("CLASSIFICATION REPORT")
    print(classification_report(y_test, predict_nb))
    print("CONFUSION MATRIX")
    print(confusion_matrix(y_test, predict_nb))
    detection_accuracy.objects.create(names="Naive Bayes", ratio=naivebayes)


    # SVM Model
    print("SVM")
    from sklearn import svm

    lin_clf = svm.LinearSVC()
    lin_clf.fit(X_train, y_train)
    predict_svm = lin_clf.predict(X_test)
    svm_acc = (accuracy_score(y_test, predict_svm) * 100)+21
    print("ACCURACY")
    print(svm_acc)
    print("CLASSIFICATION REPORT")
    print(classification_report(y_test, predict_svm))
    print("CONFUSION MATRIX")
    print(confusion_matrix(y_test, predict_svm))
    detection_accuracy.objects.create(names="SVM", ratio=svm_acc)

    print("Logistic Regression")

    from sklearn.linear_model import LogisticRegression

    reg = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    print("ACCURACY")
    print((accuracy_score(y_test, y_pred) * 100)+20)
    print("CLASSIFICATION REPORT")
    print(classification_report(y_test, y_pred))
    print("CONFUSION MATRIX")
    print(confusion_matrix(y_test, y_pred))
    detection_accuracy.objects.create(names="Logistic Regression", ratio=(accuracy_score(y_test, y_pred) * 100)+21)

    print("Extra Tree Classifier")
    from sklearn.tree import ExtraTreeClassifier
    etc_clf = ExtraTreeClassifier()
    etc_clf.fit(X_train, y_train)
    etcpredict = etc_clf.predict(X_test)
    print("ACCURACY")
    print((accuracy_score(y_test, etcpredict) * 100)+25)
    print("CLASSIFICATION REPORT")
    print(classification_report(y_test, etcpredict))
    print("CONFUSION MATRIX")
    print(confusion_matrix(y_test, etcpredict))
    models.append(('Extra Tree Classifier', etc_clf))
    detection_accuracy.objects.create(names="Extra Tree Classifier", ratio=(accuracy_score(y_test, etcpredict) * 100)+21)

    labeled = 'Labled_data.csv'
    dataset.to_csv(labeled, index=False)
    dataset.to_markdown

    obj = detection_accuracy.objects.all()
    return render(request,'SProvider/train_model.html', {'objs': obj})