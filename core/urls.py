from django.urls import path
from . import views

app_name = 'core'

urlpatterns = [
    path('', views.main_view, name='main'),
    path('page2/', views.page2_view, name='page2'),
    path('auth/', views.auth_view, name='auth'),
    path('page4/', views.page4_view, name='page4'),
    path('page5/', views.page5_view, name='page5'),
    path('page6/<int:diagnosis_id>/', views.page6_view, name='page6'),
    path('homescreen/', views.homescreen_view, name='homescreen'),

    # API Endpoints
    path('multilingual-chat/', views.multilingual_chat, name='multilingual_chat'),
    path('disaster-summarizer/', views.disaster_summarizer, name='disaster_summarizer'),
    path('crop-doctor/', views.crop_doctor, name='crop_doctor'),

    # Crop diagnosis workflow endpoints
    path('upload-crop-image/', views.upload_crop_image, name='upload_crop_image'),
    path('save-symptoms/', views.save_symptoms, name='save_symptoms'),
    path('diagnosis/<int:diagnosis_id>/', views.get_diagnosis_data, name='get_diagnosis_data'),
    path('run-analysis/<int:diagnosis_id>/', views.run_crop_analysis, name='run_crop_analysis'),
]