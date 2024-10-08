"""
URL configuration for musicgen project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
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
from generator import views
from django.conf import settings
from django.conf.urls.static import static



urlpatterns = [
    path('', views.index_page, name = 'index'),
    path('home', views.generate_music_view, name='home'),
     path('loading/', views.loading_page_view, name='loading'),
    path('generate-music', views.generate_music_async, name='generate_music_async'),
    path('generate-music-status', views.generate_music_status, name='generate_music_status'),
    path('get-random-song', views.get_random_song, name='get_random_song'),  # New route for fetching random songs
    path('music-generated/<str:file_name>/', views.music_generated_view, name='music_generated'),
    path('count-media-files', views.count_media_files, name='count_media_files'),
    path('final/', views.final_view, name='final'),  # New route for final page
    path('about', views.about_page, name='about'),


]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)