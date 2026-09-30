from faster_whisper import WhisperModel
from collections import Counter

def transcribe_audio(file_path, model):
    segments, info = model.transcribe(file_path)

    result_text = "\n".join([segment.text for segment in segments])
    return result_text


def analyze_audio(transcribed_text):
    plain_text = transcribed_text.lower().split()
    word_count = len(plain_text)
    unique_words = set(plain_text)
    unique_word_count = len(unique_words)
    word_frequency = Counter(plain_text)

    return f'The transcribed text contains {word_count} words, with {unique_word_count} unique words.\nWord frequency:\n{word_frequency.most_common(7)}'


if __name__ == "__main__":
    model = WhisperModel(
        "./models/small.en",
        device="cpu",
        compute_type="int8"
    )
    result = transcribe_audio("audiotest.m4a", model)
    analysis = analyze_audio(result)
    print(analysis)