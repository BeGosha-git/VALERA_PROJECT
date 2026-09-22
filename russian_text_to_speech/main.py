from NeuralSpeaker import NeuralSpeaker
from kairos_asr import KairosASR


asr = KairosASR()
neural_speaker = NeuralSpeaker()

def to_text(speech_file: str = "audio.wav"):
    result = asr.transcribe(wav_file=speech_file)
    print(result.full_text)

def to_speak(words: str, speaker: str = 'eugene', sample_rate: int = 48000):
    print(f'speak {words}, {speaker}, {sample_rate}')
    time_elapsed = neural_speaker.speak(words=words, speaker=speaker, save_file=False, sample_rate=sample_rate)
    print(f'Model completed in {time_elapsed} seconds')
