
import torch
from transformers import MusicgenForConditionalGeneration, AutoProcessor
from scipy.io import wavfile

def generate_music(prompt: str, file_name: str = "generated_music.wav", duration: int = 20):
    # Load the model and processor
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    
    # Set device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # Process the user input
    inputs = processor(text=[prompt], padding=True, return_tensors="pt")
    
    # Determine the number of tokens based on the desired duration
    tokens_per_second = 256 // 5  # Approximation based on 256 tokens for 5 seconds
    max_new_tokens = int(tokens_per_second * duration)
    
    # Generate audio
    audio_values = model.generate(**inputs.to(device), do_sample=True, guidance_scale=4.0, max_new_tokens=max_new_tokens)
    
    # Set sampling rate
    sampling_rate = model.config.audio_encoder.sampling_rate
    
    # Save the generated audio to a .wav file
    wavfile.write(file_name, rate=sampling_rate, data=audio_values[0].cpu().numpy())
    
    print(f"Music generated and saved to {file_name}")

if __name__ == "__main__":
    # Take user input for the text prompt
    user_input = input("Enter a description for the music: ")
    
    # Call the function to generate and save the music
    generate_music(user_input)
