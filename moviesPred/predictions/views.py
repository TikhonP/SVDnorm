from django.shortcuts import render, redirect
from gensim.summarization.summarizer import summarize


def main(request):
    if request.method == 'GET':
        return render(request, 'main.html')
    if request.method == 'POST':
        text = request.POST['text']
        vc = request.POST['wordcount']
        if vc == '':
            vc = None
        else:
            vc = int(vc)
        text = summarize(text, word_count=vc)
        return render(request, 'text.html', {'text': text})
