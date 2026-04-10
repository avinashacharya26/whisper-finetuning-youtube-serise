import whisper
import os
from pydub import AudioSegment

def transcribe_audio(file_path, language='en', model='base'):
    # Load audio file in specified format
    try:
        audio = AudioSegment.from_file(file_path)
        audio.export("temp.wav", format='wav')
        file_path = "temp.wav"
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return None

    # Load Whisper model
    model = whisper.load_model(model)

    # Transcribe audio
    result = model.transcribe(file_path, language=language)

    # Output transcription
    output_file = os.path.splitext(file_path)[0] + '_transcription.txt'
    with open(output_file, 'w') as f:
        f.write(result['text'])

    print(f'Transcription completed. Output saved to: {output_file}')