"""
URL configuration for med_chatbot_stbi project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path
from . import views

app_name = "med_chatbot_stbi"

urlpatterns = [
    path("admin/", admin.site.urls),
    path("ai/chatbot", views.get_chatbot_response, name="chatbot"),
    path("ai/chatbotpage", views.chatbot_page, name="chatbotpage"),
]


