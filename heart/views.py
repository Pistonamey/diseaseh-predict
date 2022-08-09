from django.http import HttpResponse
from django.shortcuts import render
import joblib
import numpy as np


def home(request):
    return render(request,"home.html")

def result(request):
    model = joblib.load("./heart_disease.pkl")
    values=[]
    values.append(int(request.GET['a']))
    values.append(int(request.GET['b']))
    values.append(int(request.GET['c']))
    values.append(int(request.GET['d']))
    values.append(int(request.GET['e']))
    values.append(int(request.GET['f']))
    values.append(int(request.GET['g']))
    values.append(int(request.GET['h']))
    values.append(int(request.GET['i']))
    values.append(int(request.GET['j']))
    values.append(int(request.GET['k']))
    values.append(int(request.GET['l']))
    values.append(int(request.GET['m']))
    input_data_np = np.asarray(values)
    input_data_reshaped = input_data_np.reshape(1, -1)
    ans=model.predict(input_data_reshaped)
    return render(request,"result.html",{'ans':ans})
