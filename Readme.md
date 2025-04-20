# Speech Analysis Dashboard

The Speech Analysis Dashboard app provides a comprehensive analysis of speech recordings, offering metrics related to fluency, vocal characteristics, and emotional content. The app includes features such as pitch and volume analysis, articulation rate, fluency breakdown detection, and sentiment analysis based on both transcribed text and vocal cues.

---

## Features

- **Fluency Analysis**:
  - Fluency breakdown detection system (FBDS)
  - Speech and silent segment identification
  - Syllable rate calculation
  - Statistical speech pattern analysis

- **Voice Characteristics Analysis**:
  - Pitch analysis (average, variation, range)
  - Volume analysis (intensity, variation)
  - Articulation rate measurement
  - Voice quality assessment (HNR)
  - Gender estimation

- **Tone Analysis**:
  - Emotional content understanding through vocal features
  - Sentiment analysis using transcribed text and vocal cues
  - Response tone suggestions

---

## How to Use

### 1️⃣ Clone the Repository

```commandline
git clone https://github.com/varshath-akula/SVARAG.git
cd SVARAG
```
### 2️⃣ Install FFmpeg (Required for Whisper)
- **Run the following commands in PowerShell (Windows users):**

  - Install Chocolatey:
    ```powershell
    Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))
    ```
  - After installing Chocolatey, use it to install ffmpeg:
    ```powershell
    choco install ffmpeg
    ```
- **Linux/Mac users can install FFmpeg using:**
```bash
sudo apt install ffmpeg  # Ubuntu/Debian  
brew install ffmpeg      # macOS (Homebrew)
```
### 3️⃣ Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
```
### 4️⃣ Install Dependencies
```commandline
pip install -r requirements.txt
```

### 5️⃣ Set Up API Keys
Create a .env file in the root directory and add your API keys:
```
GROQ_API_KEY = your_groq_key
```
### 6️⃣ Change the Working Directory to `app` and Run the Streamlit App
```commandline
cd app
streamlit run app.py
```