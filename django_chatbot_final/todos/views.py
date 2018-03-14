from django.shortcuts import render, redirect
from django.http import HttpResponse

from .models import Todo

def index(request):
    todos = Todo.objects.all()[:10]

    context = {
        'todos':todos
    }
    return render(request, 'index.html', context)

def details(request, id):
    todo = Todo.objects.get(id=id)

    context = {
        'todo':todo
    }
    return render(request, 'details.html', context)

from .ChatbotSolutecMaster.code_chatbot import response

def add(request):
    if(request.method == 'POST'):
        title = request.POST['title']
        text = response(request.POST['title'], '123')

        todo = Todo(title=title, text=text)
        todo.save()

        return redirect('/todos')
    else:
        return render(request, 'add.html')
