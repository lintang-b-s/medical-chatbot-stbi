from django.http.response import HttpResponse, JsonResponse
import json
from .medical_ai_chatbot import answer_pipeline
from django.shortcuts import redirect, render


def get_chatbot_response(request):
    if request.method == "POST":
        body_unicode = request.body.decode("utf-8")
        body = json.loads(body_unicode)
        question = body["question"]
        chat_history = body["chatHistory"]
        answer, context = answer_pipeline(question, chat_history)

        return JsonResponse({"chatbot_message": answer, "context": context})


def chatbot_page(request):
    return render(request, "chatbot-med.html")

