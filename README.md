# Text-to-Music Generator 🎵

This is a Django-based **Text-to-Music Generator** application capable of generating 30-second music clips based on user-provided text prompts. Additionally, the application features **Emotion-to-Music Detection**, where emotions are analyzed and converted into appropriate prompts for music generation.

---

## Features
- 🎶 **Text-to-Music Generation**: Enter a prompt, and the application generates a 30-second music clip.
- 😊 **Emotion-to-Music Detection**: Detects emotions and generates music that matches the emotional tone.
- 🎵 **Loading Music**: Plays a random loading music track while the main music is being generated.
- 📈 **Music Generation Status Tracking**: Provides real-time updates on the status of music generation.
- 🔄 **Dynamic Media Management**: Lists and serves generated and loading music tracks.

---

## Technology Stack
- **Backend**: Django, Python
- **AI Model**: MusicGen
- **Frontend**: HTML, CSS, Django Templates
- **Audio Processing**: `scipy.io.wavfile`
