from django.shortcuts import render,HttpResponse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# Create your views here.

def home(request):
   return render(request, 'index.html')


def predict(request):
    return render(request, 'predict.html')



def result(request):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression

    df = pd.read_csv(r"C:\Users\PMYLS\Downloads\archive (7)\diabetes.csv")
    x = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    clf = LogisticRegression()
    clf.fit(x_train, y_train)

    try:
        val1 = float(request.GET.get('n1'))
        val2 = float(request.GET.get('n2'))
        val3 = float(request.GET.get('n3'))
        val4 = float(request.GET.get('n4'))
        val5 = float(request.GET.get('n5'))
        val6 = float(request.GET.get('n6'))
        val7 = float(request.GET.get('n7'))
        val8 = float(request.GET.get('n8'))
    except (TypeError, ValueError):
        return render(request, 'predict.html', {'result': 'Please fill in all fields correctly.'})

    input_data = [[val1, val2, val3, val4, val5, val6, val7, val8]]
    pred = clf.predict(input_data)
    result = 'Positive' if pred[0] == 1 else 'Negative'

    return render(request, 'predict.html', {'result': result})