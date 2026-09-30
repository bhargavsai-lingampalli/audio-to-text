# 🎙️ Audio to Text

A local, private audio transcription web app built with **Streamlit**
and **faster-whisper**.

Upload an audio file or record from your microphone, transcribe it
locally, and inspect the resulting transcript with useful speech and
text analysis.

## ✨ Features

-   🎙️ **Audio transcription** using faster-whisper
-   🔒 **Local/private processing** --- audio is processed on your
    machine
-   📁 Supports multiple audio formats:
    -   MP3
    -   M4A
    -   WAV
    -   FLAC
    -   OGG
    -   AAC
    -   WMA
    -   WEBM
    -   MP4
-   🎤 **Microphone recording**
-   📦 **Batch transcription** for multiple files
-   ⏳ **Live transcription progress**
-   📝 Transcript export as `.txt`
-   🎬 Subtitle export as `.srt`
-   📋 Copy transcript, analysis, statistics, keywords, and reports
-   📊 Word-frequency analysis
-   🔑 Optional **keyword extraction** with YAKE
-   🗣️ **Filler-word detection** (`um`, `uh`, `like`, etc.)
-   ⏸️ **Pause/silence detection**
-   ⏱️ Speaking-rate and processing-speed statistics
-   🤓 Detailed **Stats for Nerds**
-   ⚙️ CPU and CUDA/GPU support
-   🧠 Configurable Whisper model path and compute type

## 🛠️ Tech Stack

-   **Python**
-   **Streamlit**
-   **faster-whisper**
-   **CTranslate2**
-   **Pandas**
-   **YAKE**

## 📋 Requirements

-   Python 3.10+ recommended
-   FFmpeg available on your system
-   Enough disk space/RAM for the selected Whisper model
-   NVIDIA GPU + CUDA is optional

## 🚀 Installation

### 1. Clone the repository

``` bash
git clone https://github.com/bhargavsai-lingampalli/audio-to-text.git
cd audio-to-text
```

### 2. Create a virtual environment

Linux/macOS:

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:

``` powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

## ▶️ Run the application

Start Streamlit with:

``` bash
streamlit run app.py
```

Then open the local URL shown in the terminal, normally:

``` text
http://localhost:8501
```

## 🧠 Whisper Model

The application is configured to use a local model path by default:

``` text
./models/small.en
```

If the model is not available at that path, the application falls back
to the `small.en` model through faster-whisper's model loading
mechanism.

You can change the model path from the application's sidebar.

### Model options

The project can be configured for different faster-whisper model sizes
depending on your hardware:

-   `tiny`
-   `base`
-   `small`
-   `medium`
-   `large-v3`

Larger models generally require more resources and take longer to
process.

## ⚙️ Application Settings

The sidebar provides options for:

### Model

-   Model path
-   Device: `cpu` or `cuda`
-   Compute type: `int8`, `float16`, or `float32`

### Display

-   Number of top words to display
-   Ignore common stopwords
-   Show timestamps in the transcript

### Optional features

-   Batch mode
-   Microphone recording
-   Keyword extraction
-   Filler-word and pause detection

## 📤 Output

For each audio file, the application provides:

### Transcript

The transcription can be:

-   Copied to the clipboard
-   Downloaded as a `.txt` file
-   Downloaded as an `.srt` subtitle file

Downloaded files use the original audio filename.

For example:

``` text
day_3.mp3
```

produces:

``` text
day_3.txt
day_3.srt
```

### Analysis

The application reports:

-   Total word count
-   Unique word count
-   Most frequent words

### Stats for Nerds

Detailed statistics include:

-   Audio duration
-   Processing time
-   Real-time factor
-   Processing speed
-   File name and size
-   Detected language
-   Model configuration
-   Number of segments
-   Average and longest segment
-   Words
-   Unique words
-   Lexical diversity
-   Character count
-   Approximate sentence count
-   Average words per sentence
-   Average word length
-   Longest word
-   Speaking rate

### Filler Words & Pauses

When enabled, the application can detect:

-   Filler words
-   Filler frequency
-   Long pauses
-   Total pause duration
-   Percentage of audio spent in detected pauses

## 📦 Batch Mode

Enable **Batch mode** to upload multiple audio files.

The application provides:

-   Individual transcripts and analysis
-   Individual `.txt` and `.srt` downloads
-   A batch summary table
-   CSV export of the batch summary

## 🔐 Privacy

The application is designed for local processing.

Audio files are processed by the application on the machine where
Streamlit is running rather than being uploaded to a third-party
transcription API.

## 📁 Project Structure

``` text
audio-to-text/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

The Whisper model files should be kept local and should **not** be
committed to Git.

## 🧪 Development

Run the application in development mode:

``` bash
streamlit run app.py
```

When debugging, `print()` output appears in the terminal running
Streamlit.

## 📌 Notes

-   Transcription speed depends on the Whisper model, CPU/GPU, compute
    type, and audio duration.
-   GPU acceleration requires a compatible CUDA environment.
-   The English-only `small.en` model is intended primarily for English
    audio.
-   For other languages, use a suitable multilingual Whisper model.

## 📄 License

This project is open source. See the repository for the applicable
license.

## 👨‍💻 Author

**Bhargav Sai Lingampalli**

GitHub: https://github.com/bhargavsai-lingampalli
