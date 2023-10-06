from django.urls import path, include
from . import views

urlpatterns = [
    path('', views.info, name='info'),
    path('', views.home, name='home'),
    path('', views.analyze_sentiment, name='analyze_sentiment'),
]