import streamlit as st
import os
import json
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from FBDSSegmentor import analyze_speech_fluency
from extract_audio_features import extract_audio_features_with_praat, get_audio_description, get_tone_instructions
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Prevent torch from being loaded initially
st.cache_resource.clear()

# More aggressive patch to prevent PyTorch path issues
import sys
import types

class NoPathModule(types.ModuleType):
    """A module that throws an exception when __path__ is accessed."""
    @property
    def __path__(self):
        raise AttributeError("This module has no __path__")

# Check if torch is already imported
if 'torch' in sys.modules:
    # If it is, patch its _classes module
    if hasattr(sys.modules['torch'], '_classes'):
        sys.modules['torch._classes'] = NoPathModule('torch._classes')
        
# Also add a monkey patch for the streamlit path watcher
def safe_extract_paths(module):
    """Safely extract paths or return an empty list if it fails."""
    try:
        if hasattr(module, '__path__'):
            if isinstance(module.__path__, list):
                return module.__path__
            elif hasattr(module.__path__, '_path'):
                try:
                    return list(module.__path__._path)
                except:
                    return []
        return []
    except:
        return []

# Try to apply the patch to Streamlit's extract_paths function
try:
    import streamlit.watcher.local_sources_watcher
    streamlit.watcher.local_sources_watcher.extract_paths = safe_extract_paths
except:
    pass

def generate_text_report(report_data):
    """Generate a formatted text report from the analysis data"""
    
    metadata = report_data["metadata"]
    fluency = report_data.get("fluency_analysis", {})
    audio = report_data.get("audio_features", {})
    
    report = f"""
==========================================================
                SPEECH ANALYSIS REPORT                    
==========================================================

ANALYSIS METADATA
----------------
File: {metadata.get('filename', 'Unknown')}
Date: {metadata.get('analysis_date', 'Unknown')}
Speaker Type: {metadata.get('speaker_type', 'Unknown')}
Context: {metadata.get('context', 'Unknown')}
Recording Quality: {metadata.get('recording_quality', 'Unknown')}

"""

    # Add audio features section if available
    if isinstance(audio, dict) and "error" not in audio:
        # Ensure all values are numeric before formatting
        duration = float(audio.get('duration', 0)) if audio.get('duration') is not None else 0
        avg_pitch = float(audio.get('avg_pitch', 0)) if audio.get('avg_pitch') is not None else 0
        pitch_std = float(audio.get('pitch_std', 0)) if audio.get('pitch_std') is not None else 0
        pitch_range = float(audio.get('pitch_range', 0)) if audio.get('pitch_range') is not None else 0
        avg_intensity = float(audio.get('avg_intensity', 0)) if audio.get('avg_intensity') is not None else 0
        intensity_variation = float(audio.get('intensity_variation', 0)) if audio.get('intensity_variation') is not None else 0
        articulation_rate = audio.get('articulation_rate')
        mean_hnr = float(audio.get('mean_hnr', 0)) if audio.get('mean_hnr') is not None else 0
        
        report += f"""
VOICE CHARACTERISTICS
------------------
Gender Estimation: {'Male' if audio.get('Gender') == 'M' else 'Female'}
Audio Duration: {duration:.2f} seconds
Average Pitch: {avg_pitch:.2f} Hz
Pitch Variation: {pitch_std:.2f}
Pitch Range: {pitch_range:.2f} Hz
Average Intensity: {avg_intensity:.2f} dB
Intensity Variation: {intensity_variation:.2f}
Articulation Rate: {articulation_rate:.2f}
Voice Quality (HNR): {mean_hnr:.2f}

VOICE DESCRIPTION
--------------
{report_data.get('audio_description', 'Not available')}
"""

    # Add fluency analysis section if available
    if isinstance(fluency, dict) and "error" not in fluency:
        report += """
FLUENCY METRICS
------------
"""
        # Add low-level metrics
        report += "Low-level Metrics:\n"
        for feature in ["rate_of_fbds_speech_segments", "std_dev_fbds_speech_segments_duration", 
                       "rate_of_fbds_silent_segments", "speech_ratio"]:
            if feature in fluency.get('features', {}):
                try:
                    value = float(fluency['features'][feature])
                    report += f"- {feature}: {value:.4f}\n"
                except (ValueError, TypeError):
                    report += f"- {feature}: {fluency['features'][feature]}\n"
        
        # Add higher-level metrics
        report += "\nHigher-level Metrics:\n"
        for feature in ["rate_of_pseudo_syllables", "std_dev_pseudo_syllable_duration", 
                       "rate_of_silent_breaks"]:
            if feature in fluency.get('features', {}):
                try:
                    value = float(fluency['features'][feature])
                    report += f"- {feature}: {value:.4f}\n"
                except (ValueError, TypeError):
                    report += f"- {feature}: {fluency['features'][feature]}\n"
        
        # Add segment counts
        if 'segment_counts' in fluency:
            report += "\nSegment Counts:\n"
            for key, value in fluency['segment_counts'].items():
                report += f"- {key}: {value}\n"
        
        # Add LLM analysis if available
        if 'llm_analysis' in fluency and fluency['llm_analysis'].get('success', False):
            report += f"""
FLUENCY ANALYSIS
-------------
{fluency['llm_analysis'].get('analysis', 'Not available')}
"""

    # Add transcribed text if available
    if "transcribed_text" in report_data:
        report += f"""
TRANSCRIBED TEXT
-------------
{report_data["transcribed_text"]}
"""

    # Add tone analysis if available
    if "tone_analysis" in report_data:
        report += f"""
RESPONSE TONE ANALYSIS
------------------
{report_data["tone_analysis"]}
"""

    report += """
==========================================================
                    END OF REPORT                       
==========================================================
"""

    return report

# Set page configuration
st.set_page_config(
    page_title="Speech Analysis Dashboard",
    page_icon="🎤",
    layout="wide"
)

# Lazy loading for Whisper to prevent initialization at startup
@st.cache_resource
def load_whisper_model():
    try:
        import whisper
        return whisper.load_model("base")  
    except Exception as e:
        st.error(f"Error loading Whisper model: {e}")
        return None

def transcribe_audio(audio_path):
    try:
        model = load_whisper_model()
        if model is None:
            return "Error: Could not load transcription model."
        result = model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        st.error(f"Transcription error: {e}")
        return f"Error: {str(e)}"

# Initialize session state for holding values between reruns
if 'transcribed_text' not in st.session_state:
    st.session_state.transcribed_text = None

# App title and description
st.title("Speech Analysis Dashboard")
st.markdown("""
This app provides comprehensive speech analysis including fluency metrics and vocal characteristics.
Upload an audio file to analyze speech patterns, fluency, and voice features.
""")

# Sidebar for settings and API key
st.sidebar.header("Settings")
api_key = st.sidebar.text_input("Groq API Key", type="password", help="Enter your Groq API key")

# Save API key to environment if provided
if api_key:
    os.environ["GROQ_API_KEY"] = api_key

# Speaker metadata input
st.sidebar.header("Speaker Information")
speaker_type = st.sidebar.selectbox(
    "Speaker Type",
    ["Native Speaker", "Non-native Speaker", "Language Learner", "Professional Speaker", "Unknown"]
)
speaking_context = st.sidebar.selectbox(
    "Speaking Context",
    ["Spontaneous Speech", "Prepared Speech", "Reading Text", "Conversation", "Unknown"]
)
recording_quality = st.sidebar.selectbox(
    "Recording Quality",
    ["High Quality", "Medium Quality", "Low Quality", "Unknown"]
)

# Navigation tabs
tab1, tab2, tab3 = st.tabs(["Upload & Analyze", "Reports", "About"])

with tab1:
    st.header("Upload Audio")
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg", "flac"])

    if uploaded_file is not None:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_filepath = tmp_file.name
        
        st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")
        
        # Transcribe button to make this optional and prevent Whisper from loading on startup
        if st.button("Transcribe Audio"):
            with st.spinner("Transcribing audio..."):
                try:
                    transcribed_text = transcribe_audio(tmp_filepath)
                    st.write("Transcribed Text:")
                    st.write(transcribed_text)
                    st.session_state.transcribed_text = transcribed_text
                except Exception as e:
                    st.error(f"Error in transcription: {str(e)}")
                    st.session_state.transcribed_text = None
        else:
            if st.session_state.transcribed_text:
                st.write("Transcribed Text:")
                st.write(st.session_state.transcribed_text)
        
        # Prepare metadata
        metadata = {
            "speaker_type": speaker_type,
            "context": speaking_context,
            "recording_quality": recording_quality
        }
        
        # Analysis button
        if st.button("Run Complete Analysis"):
            # Create a container for all analysis results
            analysis_container = st.container()
            
            with analysis_container:
                col1, col2 = st.columns(2)
                
                # First column for Fluency Analysis
                with col1:
                    with st.spinner("Analyzing speech fluency..."):
                        try:
                            # Call the analyze_speech_fluency function from FBDSSegmenter.py
                            fluency_results = analyze_speech_fluency(tmp_filepath, api_key, metadata)
                            
                            st.success("Fluency analysis complete!")
                            
                            st.header("Fluency Metrics")
                            st.subheader("Low-level Metrics")
                            metrics_df = {}
                            for feature, value in fluency_results['features'].items():
                                if feature in ["rate_of_fbds_speech_segments", "std_dev_fbds_speech_segments_duration", 
                                              "rate_of_fbds_silent_segments", "speech_ratio"]:
                                    metrics_df[feature] = [round(float(value), 4) if value is not None else None]
                            
                            st.dataframe(metrics_df)
                            
                            st.subheader("Higher-level Metrics")
                            higher_metrics_df = {}
                            for feature, value in fluency_results['features'].items():
                                if feature in ["rate_of_pseudo_syllables", "std_dev_pseudo_syllable_duration", 
                                              "rate_of_silent_breaks"]:
                                    higher_metrics_df[feature] = [round(float(value), 4) if value is not None else None]
                            
                            st.dataframe(higher_metrics_df)
                            
                            st.subheader("Segment Counts")
                            st.dataframe(fluency_results['segment_counts'])
                            
                            # LLM analysis
                            st.subheader("Fluency Analysis")
                            if fluency_results['llm_analysis']['success']:
                                st.markdown(fluency_results['llm_analysis']['analysis'])
                            else:
                                st.error(f"LLM analysis failed: {fluency_results['llm_analysis']['error']}")
                        
                        except Exception as e:
                            st.error(f"Error in fluency analysis: {str(e)}")
                            fluency_results = {"error": str(e)}
                
                # Second column for Audio Feature Analysis
                with col2:
                    with st.spinner("Extracting audio features..."):
                        try:
                            # Extract audio features
                            audio_features = extract_audio_features_with_praat(tmp_filepath)
                            # st.write(f"Audio Features : ", audio_features)
                            
                            st.success("Audio feature extraction complete!")
                            
                            # Display audio features in a nice format
                            st.header("Voice Characteristics")
                            st.write(f"**Gender Estimation:** {'Male' if audio_features['Gender'] == 'M' else 'Female'}")
                           # Safe way to format duration
                            try:
                                # duration_value = float(audio_features['duration'])
                                st.write(f"**Audio Duration:** {audio_features['duration']:.2f} seconds")
                            except (ValueError, TypeError):
                                st.write("Error in Duration")
                        
                            # Voice metrics
                            st.subheader("Voice Metrics")
                            # Define the metrics we want to display
                            metrics = [
                                ('avg_pitch', 'Average Pitch (Hz)'),
                                ('pitch_std', 'Pitch Variation'),
                                ('avg_intensity', 'Average Intensity (dB)'),
                                ('intensity_variation', 'Intensity Variation'),
                                ('articulation_rate', 'Articulation Rate'),
                                ('mean_hnr', 'Voice Quality (HNR)')
                            ]

                            # Create empty lists to store our data
                            metric_names = []
                            metric_values = []

                            # Populate the lists with careful error handling
                            for key, display_name in metrics:
                                metric_names.append(display_name)
                                
                                # Very basic handling - just display whatever is there
                                if key in audio_features:
                                    # First try to convert to float and format nicely
                                    try:
                                        value = audio_features[key]
                                        metric_values.append("{:.2f}".format(value))
                                    except (ValueError, TypeError):
                                        # If conversion fails, just show the raw value
                                        metric_values.append(str(audio_features[key]))
                                else:
                                    metric_values.append("N/A")
                                    
                            # st.write("Debug information:")
                            # for i, (key, _) in enumerate(metrics):
                            #     if key in audio_features:
                            #         st.write(f"Key: {key}, Type: {type(audio_features[key])}, Value: {repr(audio_features[key])}")                        

                            # Create the DataFrame only after all values are processed
                            metrics_df = pd.DataFrame({
                                'Metric': metric_names,
                                'Value': metric_values
                            })

                            # Display the table
                            st.table(metrics_df)
                            
                            # Voice profile visualization
                            st.subheader("Voice Profile")
                            # Prepare data for visualization
                            features = ['avg_pitch', 'pitch_std', 'avg_intensity', 
                                    'intensity_variation', 'articulation_rate', 'mean_hnr']
                            feature_names = ['Pitch', 'Pitch Var', 'Volume', 'Volume Var', 'Speech Rate', 'Voice Quality']

                            # Filter out None values and use numeric values
                            valid_features = []
                            valid_names = []
                            valid_values = []

                            for i, feat in enumerate(features):
                                # Use either numeric_features (if you created a copy) or check if the original value can be converted to float
                                try:
                                    if feat in audio_features and audio_features[feat] is not None:
                                        value = float(audio_features[feat]) if isinstance(audio_features[feat], (int, float, str)) and not audio_features[feat] in ['low', 'medium', 'high'] else None
                                        if value is not None:
                                            valid_features.append(feat)
                                            valid_names.append(feature_names[i])
                                            valid_values.append(value)
                                except (ValueError, TypeError):
                                    continue
                            
                            # Create bar chart
                            fig, ax = plt.subplots(figsize=(8, 4))
                            sns.barplot(x=valid_names, y=valid_values, palette='viridis', ax=ax)
                            plt.xticks(rotation=45)
                            plt.title('Voice Feature Profile')
                            st.pyplot(fig)
                            
                            audio_features = audio_features.copy()
                            audio_description = get_audio_description(audio_features)
                            # st.write(f"Audio_desc",audio_description)
                            # Voice description
                            st.subheader("Voice Description")
                            st.write(audio_description)
                            
                            # Get tone instructions if transcribed text is available
                            tone_instructions = None
                            if st.session_state.transcribed_text:
                                try:
                                    tone_instructions = get_tone_instructions(tmp_filepath, st.session_state.transcribed_text)
                                    st.subheader("Response Tone Analysis")
                                    st.write(tone_instructions)
                                except Exception as e:
                                    st.error(f"Error generating tone analysis: {str(e)}")
                        
                        except Exception as e:
                            st.error(f"Error in audio feature extraction: {str(e)}")
                            audio_features = {"error": str(e)}
                            audio_description = "Not available due to error."
                
                # Generate and save comprehensive report
                try:
                    # Combine all data for the report
                    report_data = {
                        "metadata": {
                            "filename": uploaded_file.name,
                            "speaker_type": speaker_type,
                            "context": speaking_context,
                            "recording_quality": recording_quality,
                            "analysis_date": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                        },
                        "fluency_analysis": fluency_results if 'fluency_results' in locals() else {"error": "Analysis failed"},
                        "audio_features": audio_features if 'audio_features' in locals() else {"error": "Analysis failed"},
                        "audio_description": audio_description if 'audio_description' in locals() else ""
                    }
                    
                    if st.session_state.transcribed_text:
                        report_data["transcribed_text"] = st.session_state.transcribed_text
                        if 'tone_instructions' in locals() and tone_instructions:
                            report_data["tone_analysis"] = tone_instructions
                    
                    # Create reports directory if it doesn't exist
                    os.makedirs("reports", exist_ok=True)
                    
                    # Save as JSON report
                    report_json = json.dumps(report_data, indent=2)
                    json_filename = f"reports/{os.path.splitext(uploaded_file.name)[0]}_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
                    with open(json_filename, "w") as f:
                        f.write(report_json)
                    
                    # Save as text report
                    text_report = generate_text_report(report_data)
                    text_filename = f"reports/{os.path.splitext(uploaded_file.name)[0]}_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt"
                    with open(text_filename, "w") as f:
                        f.write(text_report)
                    
                    # Download buttons
                    st.success("Analysis complete! Reports saved.")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="Download Full JSON Report",
                            data=report_json,
                            file_name=f"{os.path.splitext(uploaded_file.name)[0]}_analysis.json",
                            mime="application/json"
                        )
                    with col2:
                        st.download_button(
                            label="Download Text Report",
                            data=text_report,
                            file_name=f"{os.path.splitext(uploaded_file.name)[0]}_analysis.txt",
                            mime="text/plain"
                        )
                
                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")
            
            # Clean up temporary file
            if os.path.exists(tmp_filepath):
                os.unlink(tmp_filepath)
    else:
        st.info("Please upload an audio file to begin analysis.")

with tab2:
    st.header("View Previous Reports")
    
    # Check if reports directory exists
    if not os.path.exists("reports"):
        os.makedirs("reports", exist_ok=True)
    
    # List report files
    report_files = [f for f in os.listdir("reports") if f.endswith((".txt", ".json"))]
    
    if not report_files:
        st.info("No reports available. Run an analysis first.")
    else:
        selected_report = st.selectbox("Select a report to view", sorted(report_files, reverse=True))
        
        if selected_report:
            report_path = os.path.join("reports", selected_report)
            
            # Display report based on type
            if selected_report.endswith(".json"):
                with open(report_path, "r") as f:
                    report_data = json.load(f)
                
                st.json(report_data)
                
                with open(report_path, "r") as f:
                    report_content = f.read()
                
                st.download_button(
                    label="Download JSON Report",
                    data=report_content,
                    file_name=selected_report,
                    mime="application/json"
                )
            
            elif selected_report.endswith(".txt"):
                with open(report_path, "r") as f:
                    report_content = f.read()
                
                st.text_area("Report Content", report_content, height=500)
                
                st.download_button(
                    label="Download Text Report",
                    data=report_content,
                    file_name=selected_report,
                    mime="text/plain"
                )

with tab3:
    st.header("About This App")
    st.markdown("""
    ### Speech Analysis Dashboard

    This application provides comprehensive analysis of speech recordings, including:

    #### Fluency Analysis
    - Fluency breakdown detection system (FBDS)
    - Identification of speech and silent segments
    - Syllable rate calculation
    - Statistical analysis of speech patterns

    #### Voice Characteristics Analysis
    - Pitch analysis (average, variation, range)
    - Volume analysis (intensity, variation)
    - Articulation rate measurement
    - Voice quality assessment (HNR)
    - Gender estimation

    #### Tone Analysis
    - Understanding emotional content through vocal features
    - Sentiment analysis using transcribed text and vocal cues
    - Response tone suggestions

    ### How to Use
    1. Upload an audio recording
    2. Optionally transcribe the audio
    3. Click "Run Complete Analysis"
    4. View results and download reports

    ### Technical Details
    This app uses:
    - Praat for acoustic analysis
    - FBDSSegmentor for fluency analysis
    - Groq API for LLM-based analysis
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("© Speech Analysis Dashboard 2025")