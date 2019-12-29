from django.shortcuts import render, redirect
from .shrink import generate_summary

def main(request):
    if request.method == 'GET':
        return render(request, 'main.html')
    if request.method == 'POST':
        text = request.POST['text']
        vc = request.POST['wordcount']
        if vc == '':
            vc = 5
        else:
            vc = int(vc)
        text = generate_summary(text, vc)
        return render(request, 'text.html', {'text': text})
