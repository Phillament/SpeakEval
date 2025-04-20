import os
import numpy as np
import pandas as pd
import math
import torch  
_ = torch.manual_seed(0)  
import parselmouth
from parselmouth.praat import call
from groq import Groq
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from dotenv import load_dotenv

load_dotenv()

# Set environment variable to disable PyTorch's multiprocessing to avoid issues with Streamlit
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# Access the GROQ API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def extract_audio_features_with_praat(audio_path):
    """
    Extracts various audio features from a given audio file using Praat.

    Features extracted:
    - Speech rate
    - Average intensity and intensity variation (volume)
    - Average pitch and pitch variation
    - Articulation rate (speed of speech)
    - Harmonics-to-noise ratio (HNR) for voice quality
    - Speaker gender estimation (based on pitch)

    Parameters:
    audio_path (str): Path to the input audio file.

    Returns:
    dict: Dictionary containing extracted audio features.
    """
    try:
        def speech_rate(filename):
            """
            Computes the speech rate, articulation rate, and related statistics
            using intensity-based analysis.

            Parameters:
            filename (str): Path to the input audio file.

            Returns:
            dict: Dictionary containing speech rate, articulation rate,
                  and related statistics.
            """
            try:
                silencedb = -25
                mindip = 2
                minpause = 0.3
                sound = parselmouth.Sound(filename)
                originaldur = sound.get_total_duration()
                intensity = sound.to_intensity(50)
                start = call(intensity, "Get time from frame number", 1)
                nframes = call(intensity, "Get number of frames")
                end = call(intensity, "Get time from frame number", nframes)
                min_intensity = call(intensity, "Get minimum", 0, 0, "Parabolic")
                max_intensity = call(intensity, "Get maximum", 0, 0, "Parabolic")

                # get .99 quantile to get maximum (without influence of non-speech sound bursts)
                max_99_intensity = call(intensity, "Get quantile", 0, 0, 0.99)

                # estimate Intensity threshold
                threshold = max_99_intensity + silencedb
                threshold2 = max_intensity - max_99_intensity
                threshold3 = silencedb - threshold2
                if threshold < min_intensity:
                    threshold = min_intensity

                # get pauses (silences) and speakingtime
                textgrid = call(intensity, "To TextGrid (silences)", threshold3, minpause, 0.1, "silent", "sounding")
                silencetier = call(textgrid, "Extract tier", 1)
                silencetable = call(silencetier, "Down to TableOfReal", "sounding")
                npauses = call(silencetable, "Get number of rows")
                speakingtot = 0
                for ipause in range(npauses):
                    pause = ipause + 1
                    beginsound = call(silencetable, "Get value", pause, 1)
                    endsound = call(silencetable, "Get value", pause, 2)
                    speakingdur = endsound - beginsound
                    speakingtot += speakingdur

                intensity_matrix = call(intensity, "Down to Matrix")
                sound_from_intensity_matrix = call(intensity_matrix, "To Sound (slice)", 1)
                intensity_duration = call(sound_from_intensity_matrix, "Get total duration")
                intensity_max = call(sound_from_intensity_matrix, "Get maximum", 0, 0, "Parabolic")
                point_process = call(sound_from_intensity_matrix, "To PointProcess (extrema)", "Left", "yes", "no", "Sinc70")
                # estimate peak positions (all peaks)
                numpeaks = call(point_process, "Get number of points")
                t = [call(point_process, "Get time from index", i + 1) for i in range(numpeaks)]

                # fill array with intensity values
                timepeaks = []
                peakcount = 0
                intensities = []
                for i in range(numpeaks):
                    value = call(sound_from_intensity_matrix, "Get value at time", t[i], "Cubic")
                    if value > threshold:
                        peakcount += 1
                        intensities.append(value)
                        timepeaks.append(t[i])

                # fill array with valid peaks: only intensity values if preceding
                # dip in intensity is greater than mindip
                validpeakcount = 0
                if peakcount > 0:  # Check if there are any peaks
                    currenttime = timepeaks[0]
                    currentint = intensities[0]
                    validtime = []

                    for p in range(peakcount - 1):
                        following = p + 1
                        followingtime = timepeaks[p + 1]
                        dip = call(intensity, "Get minimum", currenttime, timepeaks[p + 1], "None")
                        diffint = abs(currentint - dip)
                        if diffint > mindip:
                            validpeakcount += 1
                            validtime.append(timepeaks[p])
                        currenttime = timepeaks[following]
                        currentint = call(intensity, "Get value at time", timepeaks[following], "Cubic")

                    # Look for only voiced parts
                    pitch = sound.to_pitch_ac(0.02, 30, 4, False, 0.03, 0.25, 0.01, 0.35, 0.25, 450)
                    voicedcount = 0
                    voicedpeak = []

                    for time in range(validpeakcount):
                        querytime = validtime[time]
                        whichinterval = call(textgrid, "Get interval at time", 1, querytime)
                        whichlabel = call(textgrid, "Get label of interval", 1, whichinterval)
                        value = pitch.get_value_at_time(querytime)
                        if not math.isnan(value):
                            if whichlabel == "sounding":
                                voicedcount += 1
                                voicedpeak.append(validtime[time])

                    # calculate time correction due to shift in time for Sound object versus
                    # intensity object
                    timecorrection = originaldur / intensity_duration

                    # Insert voiced peaks in TextGrid
                    call(textgrid, "Insert point tier", 1, "syllables")
                    for i in range(len(voicedpeak)):
                        position = (voicedpeak[i] * timecorrection)
                        call(textgrid, "Insert point", 1, position, "")

                    # return results
                    speakingrate = voicedcount / originaldur if originaldur > 0 else 0
                    articulationrate = voicedcount / speakingtot if speakingtot > 0 else 0
                    npause = npauses - 1
                    asd = speakingtot / voicedcount if voicedcount > 0 else 0
                    
                    return {
                        'soundname': filename,
                        'nsyll': voicedcount,
                        'npause': npause,
                        'dur(s)': originaldur,
                        'phonationtime(s)': intensity_duration,
                        'speechrate(nsyll / dur)': speakingrate,
                        "articulation rate(nsyll / phonationtime)": articulationrate,
                        "ASD(speakingtime / nsyll)": asd
                    }
                else:
                    return {
                        'soundname': filename,
                        'nsyll': 0,
                        'npause': 0,
                        'dur(s)': originaldur,
                        'phonationtime(s)': intensity_duration,
                        'speechrate(nsyll / dur)': 0,
                        "articulation rate(nsyll / phonationtime)": 0,
                        "ASD(speakingtime / nsyll)": 0
                    }
            except Exception as e:
                print(f"Error in speech_rate: {e}")
                return {
                    'soundname': filename,
                    'nsyll': 0,
                    'npause': 0,
                    'dur(s)': 0,
                    'phonationtime(s)': 0,
                    'speechrate(nsyll / dur)': 0,
                    "articulation rate(nsyll / phonationtime)": 0,
                    "ASD(speakingtime / nsyll)": 0
                }

        # Load the audio file into a Sound object
        snd = parselmouth.Sound(audio_path)

        # Extract the total duration of the audio file in seconds
        duration = snd.get_total_duration()

        # Convert the sound to an intensity object with a time step of 100 ms
        intensity = call(snd, "To Intensity", 100, 0.0, True)
        # Calculate the average intensity over the entire duration using energy averaging
        avg_intensity = call(intensity, "Get mean", 0, 0, "energy")
        # Volume variation
        intensity_values = intensity.values
        volume_variation = np.std(intensity_values)

        # Convert the sound to a pitch object with a time step of 0.0 (auto), min pitch 75 Hz, max pitch 600 Hz
        pitch = call(snd, "To Pitch", 0.0, 75, 600)
        # Calculate the average pitch over the entire duration in Hertz
        avg_pitch = call(pitch, "Get mean", 0, 0, "Hertz")

        Gender = "M" if avg_pitch < 165 else "F"

        # Pitch variation
        pitch_values = pitch.selected_array['frequency']
        pitch_values = pitch_values[pitch_values != 0]  # Remove unvoiced frames
        pitch_std = np.std(pitch_values) if len(pitch_values) > 0 else 0
        pitch_range = np.ptp(pitch_values) if len(pitch_values) > 0 else 0

        try:
            articulation_rate = speech_rate(audio_path)['articulation rate(nsyll / phonationtime)']
        except:
            articulation_rate = None

        # Voice quality (using harmonics-to-noise ratio as a proxy)
        try:
            harmonicity = snd.to_harmonicity()
            mean_hnr = call(harmonicity, "Get mean", 0, 0)
        except:
            mean_hnr = 0

        return {
            "Gender": Gender,
            "duration": duration,
            "avg_intensity": avg_intensity,
            "intensity_variation": volume_variation,
            "avg_pitch": avg_pitch,  # Hertz
            "pitch_std": pitch_std,
            "pitch_range": pitch_range,
            "articulation_rate": articulation_rate,  # number of syllables per second
            "mean_hnr": mean_hnr,
        }
    except Exception as e:
        print(f"Error in extract_audio_features_with_praat: {e}")
        return {
            "Gender": "M",  # Default value
            "duration": 0,
            "avg_intensity": 0,
            "intensity_variation": 0,
            "avg_pitch": 0,
            "pitch_std": 0,
            "pitch_range": 0,
            "articulation_rate": 0,
            "mean_hnr": 0,
        }


def get_audio_description(audio_features):
    """
    Generates a human-readable description of the speaker's vocal characteristics
    based on extracted audio features.

    Parameters:
    audio_features (dict): Dictionary containing extracted audio features.

    Returns:
    str: A descriptive analysis of the speaker's voice.
    """
    features_copy = audio_features.copy()
    try:
        def get_categories(df):
            """
            Categorizes a feature value based on predefined thresholds

            Parameters:
                 df(dict) : Dictionary containing the extracted audio features

            Returns:
                dict : Dictionary with the Audio Features Categorized as low, medium and high
            """
            thresholds = {'M': {'duration': {'low': 2.40296875, 'medium_high': 6.2399375},
                                'avg_intensity': {'low': 54.084510358584424,
                                                  'medium_high': 63.877408664419114},
                                'intensity_variation': {'low': 7.867059257814136,
                                                        'medium_high': 10.656916311840614},
                                'avg_pitch': {'low': 128.2025173178583, 'medium_high': 187.9752824882641},
                                'pitch_std': {'low': 32.04305854178459, 'medium_high': 79.28567646277048},
                                'pitch_range': {'low': 137.01219349653715, 'medium_high': 435.1436189986501},
                                'articulation_rate': {'low': 3.223531782008662,
                                                      'medium_high': 4.587496107054758},
                                'mean_hnr': {'low': 6.961254688122402, 'medium_high': 9.929355905136154}},
                          'F': {'duration': {'low': 2.28, 'medium_high': 5.2},
                                'avg_intensity': {'low': 52.68011376219538,
                                                  'medium_high': 64.14112903043477},
                                'intensity_variation': {'low': 7.516745097145313,
                                                        'medium_high': 10.591423940893918},
                                'avg_pitch': {'low': 204.08844778422682, 'medium_high': 270.6851331889209},
                                'pitch_std': {'low': 42.41832606902338, 'medium_high': 81.21318451531326},
                                'pitch_range': {'low': 190.37198363829287, 'medium_high': 410.5080184627136},
                                'articulation_rate': {'low': 3.42500255313981,
                                                      'medium_high': 4.741379310344827},
                                'mean_hnr': {'low': 8.190640365419767, 'medium_high': 11.826573007440683}}}

            def categorize(value, thresholds):
                """
                Categorizes a feature value based on predefined thresholds.

                Parameters:
                    value(float) : Value of the feature
                    thresholds(dict) : Gender wise thresholds for the audio features

                Returns:
                    str : The category in which the audio feature's value lies
                """
                if pd.isna(value) or value == 0:
                    return 'none'
                if value <= thresholds['low']:
                    return 'low'
                elif value <= thresholds['medium_high']:
                    return 'medium'
                else:
                    return 'high'

            features = list(thresholds['M'].keys())
            gender = df.get('Gender', 'M')  # Default to M if not present
            if gender not in ['M', 'F']:
                gender = 'M'  # Default to M if invalid gender
                
            for feature in features:
                if feature in df and feature in thresholds[gender]:
                    df[feature] = categorize(df[feature], thresholds[gender][feature])
            return df

        def generate_impression(df):
            """
            Creates a natural language impression of the speaker's tone and style.

            Parameters:
                 df(dict) : Dictionary containing the categorized audio features

            Returns:
                str : A comprehensive impression about the speaker's pitch, volume and articulation rate
            """
            pitch = df.get('avg_pitch', 'medium')
            pitch_var = df.get('pitch_std', 'medium')
            volume = df.get('avg_intensity', 'medium')
            volume_var = df.get('intensity_variation', 'medium')
            rate = df.get('articulation_rate', 'medium')
            
            # Pitch impression
            if pitch in ['high']:
                pitch_impression = "uses a higher pitch"
            elif pitch in ['low']:
                pitch_impression = "uses a lower pitch"
            else:
                pitch_impression = "has a moderate pitch"

            if pitch_var in ['high']:
                pitch_impression += " with noticeable variation, suggesting expressiveness"
            elif pitch_var in ['low']:
                pitch_impression += " that remains steady, potentially indicating calmness or seriousness"
            else:
                pitch_impression += " with typical variation"

            # Volume impression
            if volume in ['high']:
                volume_impression = "speaking loudly, which might indicate excitement, confidence, or urgency"
            elif volume in ['low']:
                volume_impression = "speaking softly, possibly suggesting calmness, shyness, or caution"
            else:
                volume_impression = "using a moderate volume"

            if volume_var in ['high']:
                volume_impression += ", with significant volume changes"
            elif volume_var in ['low']:
                volume_impression += ", with little volume variation"
            else:
                volume_impression += ", with normal volume variation"

            # Speech rate impression
            if rate in ['high']:
                rate_impression = "talking quickly, which could indicate excitement, urgency, or nervousness"
            elif rate in ['low']:
                rate_impression = "talking slowly, possibly suggesting thoughtfulness, hesitation, or calmness"
            else:
                rate_impression = "speaking at a moderate pace"

            # Combine impressions into a single, flowing sentence
            impression = f"The target speaker {pitch_impression}, while {volume_impression}, and is {rate_impression}."

            return impression

        categorized_features = get_categories(features_copy)
        impression = generate_impression(categorized_features)
        return impression
    except Exception as e:
        print(f"Error in get_audio_description: {e}")
        return "Unable to generate audio description due to insufficient or invalid audio features."

def load_whisper_model():
    try:
        import whisper
        return whisper.load_model("base")  # try "small", "medium", or "large" for better accuracy
    except Exception as e:
        return None
    
def transcribe_audio(audio_path):
    try:
        model = load_whisper_model()
        if model is None:
            return "Error: Could not load transcription model."
        result = model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        return f"Error: {str(e)}"


def get_response(audio_path,transcribed_text, audio_description):
    """
    Generates response structuring instructions based on transcribed text and vocal tone.

    Parameters:
    transcribed_text (str): The textual transcription of the audio input.
    audio_description (str): The description of the speaker's vocal characteristics.

    Returns:
    str: Instructions on how to structure the response based on sentiment and emotion.
    """
    try:
        transcribed_text = transcribe_audio(audio_path)
        client = Groq(api_key=GROQ_API_KEY)
        
        # Fix: Format the system message with the actual values before sending
        system_message = """
        You are an expert in sentiment and emotional analysis. Analyze the sentiment of the
        input query by considering both the transcribed text and the tone inferred from the audio
        description. Identify the emotional tone (e.g., Nervousness, Confidence, Calmness etc.)
        and categorize the sentiment into one or more of the following categories: Positive, Negative,
        Neutral, Urgent, Concerned, or any other relevant category. Based on the detected sentiment, provide
        instructions on how to structure the response. 
        
        Important: Do not provide any instructions or content related to the actual output content of the query 
        itself. Output only the instructions for structuring the tone of the response. Do not provide any examples for
        the output.
        Transcribed text : {Transcribed_text}

        Audio Description : {Audio_Description} (In terms of pitch, volume, variation in pitch, variation in volume, articulation rate)

        Example Query:

        Transcribed Text: "I've been taking the steroids you prescribed, but I'm feeling dizzy. Is this normal?"
        Audio Description: "The pitch is slightly higher with some variation. Volume is moderate, with slight 
        fluctuations, and the articulation rate is slightly faster than usual."

        Example Output:
        The response should adopt an empathetic tone, acknowledging the speaker's concern. Given the slightly
        higher pitch and faster articulation rate, this suggests mild urgency or anxiety. The tone should remain
        reassuring yet clear, offering practical advice to help the speaker feel supported. The tone should
        avoid sounding dismissive or overly technical and should be compassionate.
        """
        
        # Replace the placeholders with actual values
        system_message = system_message.replace("{Transcribed_text}", transcribed_text)
        system_message = system_message.replace("{Audio_Description}", audio_description)
        
        stream = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user",
                    "content":
                        f"""
                    Transcribed text = {transcribed_text}
                    Audio Description = {audio_description}
                    """
                }
            ],
            temperature=0.5,
            max_completion_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
        )
        return stream.choices[0].message.content
    except Exception as e:
        print(f"Error in get_response: {e}")
        return "Unable to generate response instructions. Please check your Groq API key and connection."

def get_tone_instructions(audio_file_path, transcribed_text):
    """
    Generates response structuring instructions based on both transcribed text and path to the audio file.

    This function extracts audio features from the given audio file, generates a descriptive
    summary of the vocal characteristics, and then uses both the transcribed text and
    audio description to derive tone-based response instructions.

    Parameters:
    audio_file_path (str): The path to the audio file to be analyzed.
    transcribed_text (str): The textual transcription of the audio content.

    Returns:
    str: Instructions on how to structure the response based on sentiment and vocal tone.
    """
    try:
        # Check if the audio file exists
        if not os.path.exists(audio_file_path):
            return "Error: Audio file not found."
            
        # Extract audio features
        audio_features = extract_audio_features_with_praat(audio_file_path)
        print(audio_features)
        
        # Generate audio description
        audio_description = get_audio_description(audio_features)
        print(audio_description)
        
        # Generate response guidelines
        transcribed_text = transcribe_audio(audio_file_path)
        tone_instructions = get_response(audio_file_path,transcribed_text, audio_description)
        print(tone_instructions)
        
        return tone_instructions
    except Exception as e:
        print(f"Error in get_tone_instructions: {e}")
        return f"Error generating tone instructions: {str(e)}"
     
# transcribed_text = transcribe_audio(r"E:/Fluency_Report/Research_paper_audio.wav")
# print(get_tone_instructions((r"E:/Fluency_Report/Research_paper_audio.wav"),transcribed_text))

