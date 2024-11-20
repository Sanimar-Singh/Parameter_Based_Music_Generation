from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
import time
import random
import os
import torch
from transformers import MusicgenForConditionalGeneration, AutoProcessor
from scipy.io import wavfile
import cv2
from deepface import DeepFace
import logging
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
import numpy as np
import base64
from deepface import DeepFace
import json
import google.generativeai as genai



# Load model once when the Django app starts
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model.to(device)

def index_page(request):
    return render(request, 'generator/index.html')

def about_page(request):
    return render(request, 'generator/about.html')

def generate_music_view(request):
    # This is the view that should render the home page
    if request.method == 'POST':
        prompt = request.POST.get('prompt')
        return render(request, 'generator/loading.html', {'prompt': prompt})
    
    # Render the form page on GET request
    return render(request, 'generator/home.html')

# Track music generation status (this can be more sophisticated with a task queue if necessary)
# generation_status = {'done': False, 'file_name': ''}
generation_status = {'done': False, 'file_name': '', 'prompt': '', 'emotion': ''}


def loading_page_view(request):
    # Path to the directory containing your loading songs
    music_directory = os.path.join('media', 'loading_music')

    # List all the files in the directory (make sure they are audio files)
    songs = [f for f in os.listdir(music_directory) if f.endswith('.mp3') or f.endswith('.wav')]

    # Pick a random song from the list of songs
    if songs:
        random_song = random.choice(songs)
        random_song_url = f'/media/loading_music/{random_song}'  # Create the URL to the song
    else:
        random_song_url = None  # Handle the case where no files are available

    return render(request, 'generator/loading.html', {'random_song_url': random_song_url})

def get_random_song(request):
    # Path to the directory containing your loading songs
    music_directory = os.path.join('media', 'loading_music')

    # List all the files in the directory (make sure they are audio files)
    songs = [f for f in os.listdir(music_directory) if f.endswith('.mp3') or f.endswith('.wav')]

    # Pick a random song from the list of songs
    if songs:
        random_song = random.choice(songs)
        random_song_url = f'/media/loading_music/{random_song}'
        return JsonResponse({'random_song_url': random_song_url})
    else:
        return JsonResponse({'random_song_url': None})

# Asynchronous music generation view
# def generate_music_async(request):
#     if request.method == 'POST':
#         prompt = request.POST.get('prompt')
#         logger.info(f"Received prompt: {prompt}")
        
#         if prompt:
#             # Log music generation starting
#             logger.info(f"Starting music generation for prompt: {prompt}")
#             file_name = generate_music(prompt)

#             if file_name:
#                 logger.info(f"Music generated successfully: {file_name}")
#                 global generation_status
#                 generation_status = {'done': True, 'file_name': file_name}
#                 return JsonResponse({'status': 'success', 'file_name': file_name})
#             else:
#                 logger.error("Music generation failed.")
#                 return JsonResponse({'status': 'failed', 'message': 'Music generation failed'}, status=500)
#     return JsonResponse({'status': 'failed', 'message': 'Invalid request'}, status=400)

generation_status = {'done': False, 'file_name': '', 'prompt': ''}

def generate_music_async(request):
    if request.method == 'POST':
        prompt = request.POST.get('prompt')

        if prompt:
            file_name = generate_music(prompt)  # Generates music
            if file_name:
                global generation_status
                generation_status = {
                    'done': True,
                    'file_name': file_name,
                    'prompt': prompt  # Store the prompt to display later
                }
                return JsonResponse({'status': 'success'})
            else:
                return JsonResponse({'status': 'failed', 'message': 'Music generation failed'}, status=500)
    return JsonResponse({'status': 'failed', 'message': 'Invalid request'}, status=400)


# Check the status of the music generation
generation_status = {'done': False, 'file_name': ''}

def generate_music_status(request):
    # Logging to check what the status is at the time of the request
    logging.info(f"Music generation status: {generation_status}")
    
    # Return the status of the music generation
    return JsonResponse(generation_status)

def music_generated_view(request):
    # List the media directory and get the latest generated file
    media_directory = os.path.join('media')
    files = [f for f in os.listdir(media_directory) if os.path.isfile(os.path.join(media_directory, f))]
    
    # Assuming the latest file is the newly generated one (based on filename or timestamp)
    latest_file = max(files, key=lambda f: os.path.getctime(os.path.join(media_directory, f)))

    # Construct the URL to the latest music file
    music_file_url = f'/media/{latest_file}'

    # Render the home page template and pass the music file URL to it
    return render(request, 'generator/home.html', {'music_file_url': music_file_url})


# Music generation function
def generate_music(prompt: str, duration: int = 20):
    # Process to generate the music file
    inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(device)

    tokens_per_second = 256 // 5
    max_new_tokens = int(tokens_per_second * duration)
    audio_values = model.generate(**inputs, do_sample=True, guidance_scale=4.0, max_new_tokens=max_new_tokens)

    # Save the generated audio to a WAV file in the media directory
    file_name = f'generated_music_{random.randint(1000, 9999)}.wav'
    file_path = os.path.join('media', file_name)
    wavfile.write(file_path, 16000, audio_values.cpu().numpy())
    
    return file_name

def count_media_files(request):
    media_directory = os.path.join('media')
    files = [f for f in os.listdir(media_directory) if os.path.isfile(os.path.join(media_directory, f))]
    return JsonResponse({'file_count': len(files)})

def final_view(request):
    global generation_status
    # Fetch the prompt and the generated file from the generation_status
    prompt = generation_status['prompt']
    file_name = generation_status['file_name']

    music_file_url = f'/media/{file_name}'
    return render(request, 'generator/final.html', {'prompt': prompt, 'music_file_url': music_file_url})


def emotion_view(request):
    """Render the emotion detection page"""
    return render(request, 'generator/emotion.html')
    # return render(request, 'emotion.html')

# @csrf_exempt
# def detect_emotion(request):
#     if request.method == 'POST':
#         try:
#             # Get the image data from the POST request
#             image_data = json.loads(request.body)['image']
#             # Remove the "data:image/jpeg;base64," header
#             image_data = image_data.split(',')[1]
            
#             # Decode base64 image
#             image_bytes = base64.b64decode(image_data)
#             image_array = np.frombuffer(image_bytes, dtype=np.uint8)
#             frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
#             # Convert frame to RGB (DeepFace expects RGB)
#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
#             # Perform emotion analysis
#             result = DeepFace.analyze(rgb_frame, actions=['emotion'], enforce_detection=False)
#             emotion = result[0]['dominant_emotion']
            
#             return JsonResponse({'emotion': emotion})
#         except Exception as e:
#             return JsonResponse({'error': str(e)}, status=500)
    
#     return JsonResponse({'error': 'Invalid request method'}, status=400)

genai.configure(api_key='AIzaSyCGFbnuJG47vu5q0X7afsy66y0aUimv8mM')
model = genai.GenerativeModel('gemini-pro')

def generate_music_prompts(emotion):
    """
    Generate dynamic music prompts using Gemini API based on the detected emotion
    """
    prompt_template = f"""
    Generate 5 unique and creative prompts for music generation based on the emotion: {emotion}.
    Each prompt should:
    1. Be specific about musical elements (tempo, instruments, mood, genre)
    2. Be descriptive and evocative
    3. Include atmospheric or contextual details
    4. Be suitable for AI music generation
    5. Be 1-2 sentences long
    
    Format each prompt on a new line and make them diverse in style and genre.
    Don't number them or add any prefixes.
    """
    
    try:
        response = model.generate_content(prompt_template)
        # Split the response into individual prompts and clean them
        prompts = [prompt.strip() for prompt in response.text.split('\n') if prompt.strip()]
        # Take only the first 5 prompts if more are generated
        return prompts[:5]
    except Exception as e:
        print(f"Error generating prompts: {str(e)}")
        # Fallback prompts in case of API error
        return [
            f"Create an emotive {emotion} piece with piano and strings",
            f"Generate a {emotion} soundtrack with atmospheric elements",
            f"Compose a {emotion} melody with modern instruments",
            f"Design a {emotion} ambient track with nature sounds",
            f"Produce a {emotion} fusion piece with world music elements"
        ]

@csrf_exempt
def detect_emotion(request):
    if request.method == 'POST':
        try:
            # Get the image data from the POST request
            image_data = json.loads(request.body)['image']
            # Remove the "data:image/jpeg;base64," header
            image_data = image_data.split(',')[1]
            
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            # Convert frame to RGB (DeepFace expects RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Perform emotion analysis
            result = DeepFace.analyze(rgb_frame, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']
            
            # Generate dynamic music prompts using Gemini
            music_prompts = generate_music_prompts(emotion)
            
            return JsonResponse({
                'emotion': emotion,
                'music_prompts': music_prompts
            })
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=400)