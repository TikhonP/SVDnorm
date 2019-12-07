from django.http import HttpResponse
from django.shortcuts import render, redirect

def main(request):
    if request.method == 'GET':
        return render(request, 'main.html')