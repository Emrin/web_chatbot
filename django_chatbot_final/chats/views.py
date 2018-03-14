from django.shortcuts import render, redirect
from django.http import HttpResponse

from .models import Chat

def index(request):
    chats = Chat.objects.all()[:10]

    context = {
        'chats':chats
    }
    return render(request, 'index.html', context)

def details(request, id):
    chat = Chat.objects.get(id=id)

    context = {
        'todo':chat
    }
    return render(request, 'details.html', context)

from .ChatbotSolutecMaster.code_chatbot import response

def add(request):
    if(request.method == 'POST'):
        title = request.POST['title']
        text = response(request.POST['title'], '123', False)

        chat = Chat(title=title, text=text)
        chat.save()

        return redirect('/chats')
    else:
        return render(request, 'add.html')
