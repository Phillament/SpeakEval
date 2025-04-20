import numpy as np
import librosa
import scipy.signal
from sklearn.mixture import GaussianMixture
import json
import os
import requests
from groq import Groq

class FBDSSegmenter:
    """
    Frame-Based Descriptor Segmentation (FBDS) algorithm for speech segmentation.
    """
    def __init__(self, short_window_size=0.025, long_window_size=0.15, step_size=0.01, 
                 threshold=0.5, ar_order=10):
        """
        Initialize the FBDS segmenter.
        
        Parameters:
        -----------
        short_window_size : float
            Size of the short-term analysis window in seconds
        long_window_size : float
            Size of the longer-term buffer window in seconds
        step_size : float
            Step size for moving the short window in seconds
        threshold : float
            Threshold for the Kullback-Leibler divergence to detect boundaries
        ar_order : int
            Order of the autoregressive model
        """
        self.short_window_size = short_window_size
        self.long_window_size = long_window_size
        self.step_size = step_size
        self.threshold = threshold
        self.ar_order = ar_order
    
    def _fit_ar_model(self, signal):
        """Fit an autoregressive Gaussian model to the signal."""
        # Compute AR coefficients using Burg's method
        # The lpc function returns only the AR coefficients, not additional values
        ar_coeffs = librosa.core.lpc(signal, order=self.ar_order)
        
        # Estimate the variance of the residual
        residual = scipy.signal.lfilter(ar_coeffs, 1, signal)
        variance = np.var(residual)
        
        return ar_coeffs, variance
    
    def _compute_kl_divergence(self, ar_model1, ar_model2):
        """
        Calculate Kullback-Leibler divergence between two AR Gaussian models.
        Using a robust approach that handles zero or very small variance.
        
        Parameters:
        -----------
        ar_model1, ar_model2 : tuple
            Each tuple contains (ar_coefficients, variance)
        
        Returns:
        --------
        float
            KL divergence value
        """
        ar_coeffs1, var1 = ar_model1
        ar_coeffs2, var2 = ar_model2
        
        # Add a small regularization term to avoid division by zero
        epsilon = 1e-10
        var1 = max(var1, epsilon)
        var2 = max(var2, epsilon)
        
        # Compute KL divergence for Gaussian distributions with regularization
        try:
            kl_div = np.log(var2 / var1) + (var1 + (ar_coeffs1[0] - ar_coeffs2[0])**2) / var2 - 1
        except (ZeroDivisionError, RuntimeWarning):
            # Fallback if computation fails
            kl_div = 0.1  # Default small value
    
        # For higher-order AR models, add term based on coefficient differences
        if len(ar_coeffs1) > 1 and len(ar_coeffs2) > 1:
            # Take only the coefficients after the first one
            a1 = ar_coeffs1[1:] if len(ar_coeffs1) > 1 else np.array([0])
            a2 = ar_coeffs2[1:] if len(ar_coeffs2) > 1 else np.array([0])
            
            # Make sure they have the same length
            max_len = max(len(a1), len(a2))
            a1_padded = np.pad(a1, (0, max_len - len(a1)), 'constant')
            a2_padded = np.pad(a2, (0, max_len - len(a2)), 'constant')
            
            # Add a term based on coefficient differences with safety check
            coeff_diff = np.sum((a1_padded - a2_padded)**2)
            try:
                kl_div += coeff_diff / (2 * var2)
            except (ZeroDivisionError, RuntimeWarning):
                kl_div += coeff_diff * 5  # Alternative calculation that avoids division
        
        # Ensure KL divergence is non-negative and not NaN
        kl_div = 0 if np.isnan(kl_div) else kl_div
        return max(0, kl_div)
    
    def segment_forward(self, signal, sr):
        """
        Perform forward segmentation using FBDS.
        
        Parameters:
        -----------
        signal : numpy.ndarray
            The audio signal
        sr : int
            Sample rate
        
        Returns:
        --------
        list
            List of boundary indices
        """
        short_window_samples = int(self.short_window_size * sr)
        long_window_samples = int(self.long_window_size * sr)
        step_samples = int(self.step_size * sr)
        
        boundaries = [0]  # Start with beginning of signal
        long_buffer_start = 0
        
        for i in range(0, len(signal) - short_window_samples, step_samples):
            # Extract the short window
            short_window = signal[i:i+short_window_samples]
            
            # Extract the long buffer
            long_buffer = signal[long_buffer_start:i+short_window_samples]
            
            # If both windows have enough data
            if len(short_window) == short_window_samples and len(long_buffer) > self.ar_order:
                # Fit AR models
                short_model = self._fit_ar_model(short_window)
                long_model = self._fit_ar_model(long_buffer)
                
                # Calculate KL divergence
                kl_div = self._compute_kl_divergence(short_model, long_model)
                
                # If KL divergence exceeds threshold, mark a boundary
                if kl_div > self.threshold:
                    boundaries.append(i)
                    long_buffer_start = i  # Reset long buffer
        
        # Add end of signal as final boundary
        if boundaries[-1] != len(signal):
            boundaries.append(len(signal))
            
        return boundaries
    
    def segment_backward(self, signal, sr):
        """Perform backward segmentation (same as forward but on reversed signal)."""
        reversed_signal = np.flip(signal)
        reversed_boundaries = self.segment_forward(reversed_signal, sr)
        
        # Convert reversed boundaries to original signal indices
        boundaries = [len(signal) - b for b in reversed_boundaries]
        return sorted(boundaries)
    
    def segment(self, signal, sr):
        """
        Segment the signal using FBDS with both forward and backward passes.
        
        Parameters:
        -----------
        signal : numpy.ndarray
            The audio signal
        sr : int
            Sample rate
        
        Returns:
        --------
        list of tuples
            List of (start_idx, end_idx) tuples representing segments
        """
        forward_boundaries = self.segment_forward(signal, sr)
        backward_boundaries = self.segment_backward(signal, sr)
        
        # Merge boundaries from both directions and remove duplicates
        all_boundaries = sorted(set(forward_boundaries + backward_boundaries))
        
        # Convert to segments
        segments = []
        for i in range(len(all_boundaries) - 1):
            segments.append((all_boundaries[i], all_boundaries[i+1]))
        
        return segments

class SpeechFluencyPredictor:
    """
    Predicts speech fluency by computing low-level and higher-level features.
    """
    def __init__(self, silence_energy_threshold=0.04, silence_duration_threshold=0.25, 
                 energy_ratio_threshold=0.7):
        """
        Initialize the speech fluency predictor.
        
        Parameters:
        -----------
        silence_energy_threshold : float
            Ratio of maximum energy to classify a segment as silence
        silence_duration_threshold : float
            Minimum duration (in seconds) to consider a silence as a silent break
        energy_ratio_threshold : float
            Maximum energy decrease ratio to consider consecutive segments as part of the same pseudo-syllable
        """
        self.silence_energy_threshold = silence_energy_threshold
        self.silence_duration_threshold = silence_duration_threshold
        self.energy_ratio_threshold = energy_ratio_threshold
        self.fbds_segmenter = FBDSSegmenter()
    
    def _compute_segment_energy(self, signal, start_idx, end_idx):
        """Compute the maximum energy of a segment."""
        segment = signal[start_idx:end_idx]
        return np.max(np.abs(segment))
    
    def process(self, audio_file):
        """
        Process an audio file to extract speech fluency predictors.
        
        Parameters:
        -----------
        audio_file : str
            Path to the audio file
        
        Returns:
        --------
        dict
            Dictionary containing both low-level and higher-level predictors
        """
        # Load audio file
        signal, sr = librosa.load(audio_file, sr=None)
        
        # Step 1: Apply FBDS segmentation
        fbds_segments = self.fbds_segmenter.segment(signal, sr)
        
        # Step 2: Classify segments as speech or silence
        max_energy = np.max(np.abs(signal))
        silence_threshold = max_energy * self.silence_energy_threshold
        
        fbds_speech_segments = []
        fbds_silent_segments = []
        
        for start_idx, end_idx in fbds_segments:
            segment_energy = self._compute_segment_energy(signal, start_idx, end_idx)
            segment_duration = (end_idx - start_idx) / sr
            
            if segment_energy > silence_threshold:
                fbds_speech_segments.append((start_idx, end_idx, segment_duration))
            else:
                fbds_silent_segments.append((start_idx, end_idx, segment_duration))
        
        # Step 3: Cluster segments into pseudo-syllables and silent breaks
        pseudo_syllables = []
        current_syllable = []
        
        # Cluster speech segments into pseudo-syllables
        for i, (start_idx, end_idx, duration) in enumerate(fbds_speech_segments):
            if not current_syllable:
                current_syllable = [(start_idx, end_idx, duration)]
            else:
                prev_segment = current_syllable[-1]
                prev_energy = self._compute_segment_energy(signal, prev_segment[0], prev_segment[1])
                curr_energy = self._compute_segment_energy(signal, start_idx, end_idx)
                
                # Check if the energy decrease is within threshold
                if curr_energy >= self.energy_ratio_threshold * prev_energy:
                    current_syllable.append((start_idx, end_idx, duration))
                else:
                    # Complete the current syllable and start a new one
                    syllable_start = current_syllable[0][0]
                    syllable_end = current_syllable[-1][1]
                    syllable_duration = (syllable_end - syllable_start) / sr
                    pseudo_syllables.append((syllable_start, syllable_end, syllable_duration))
                    current_syllable = [(start_idx, end_idx, duration)]
            
            # Handle the last segment
            if i == len(fbds_speech_segments) - 1 and current_syllable:
                syllable_start = current_syllable[0][0]
                syllable_end = current_syllable[-1][1]
                syllable_duration = (syllable_end - syllable_start) / sr
                pseudo_syllables.append((syllable_start, syllable_end, syllable_duration))
        
        # Cluster silent segments into silent breaks
        silent_breaks = []
        current_silence = []
        
        for i, (start_idx, end_idx, duration) in enumerate(fbds_silent_segments):
            if not current_silence:
                current_silence = [(start_idx, end_idx, duration)]
            elif current_silence[-1][1] == start_idx:  # Check if contiguous
                current_silence.append((start_idx, end_idx, duration))
            else:
                # Complete the current silence and start a new one
                silence_start = current_silence[0][0]
                silence_end = current_silence[-1][1]
                silence_duration = (silence_end - silence_start) / sr
                
                if silence_duration > self.silence_duration_threshold:
                    silent_breaks.append((silence_start, silence_end, silence_duration))
                
                current_silence = [(start_idx, end_idx, duration)]
            
            # Handle the last segment
            if i == len(fbds_silent_segments) - 1 and current_silence:
                silence_start = current_silence[0][0]
                silence_end = current_silence[-1][1]
                silence_duration = (silence_end - silence_start) / sr
                
                if silence_duration > self.silence_duration_threshold:
                    silent_breaks.append((silence_start, silence_end, silence_duration))
        
        # Compute duration of recording
        total_duration = len(signal) / sr
        
        # Compute low-level predictors
        low_level_predictors = {
            "rate_of_fbds_speech_segments": len(fbds_speech_segments) / total_duration,
            "std_dev_fbds_speech_segments_duration": np.std([d for _, _, d in fbds_speech_segments]),
            "rate_of_fbds_silent_segments": len(fbds_silent_segments) / total_duration,
            "speech_ratio": sum(d for _, _, d in fbds_speech_segments) / total_duration
        }
        
        # Compute higher-level predictors
        higher_level_predictors = {
            "rate_of_pseudo_syllables": len(pseudo_syllables) / total_duration,
            "std_dev_pseudo_syllable_duration": np.std([d for _, _, d in pseudo_syllables]),
            "rate_of_silent_breaks": len(silent_breaks) / total_duration
        }
        
        # Combine all predictors
        all_predictors = {**low_level_predictors, **higher_level_predictors}
        
        return all_predictors, {
            "fbds_segments": fbds_segments,
            "fbds_speech_segments": fbds_speech_segments,
            "fbds_silent_segments": fbds_silent_segments,
            "pseudo_syllables": pseudo_syllables,
            "silent_breaks": silent_breaks
        }

class FluencyAnalysisLLM:
    """
    Uses Groq LLM API to analyze speech fluency metrics and provide human-interpretable insights.
    """
    def __init__(self, api_key=None, model="llama3-70b-8192"):
        """
        Initialize the FluencyAnalysisLLM.
        
        Parameters:
        -----------
        api_key : str
            API key for the Groq LLM service
        model : str
            Model identifier to use (default: llama3-70b-8192)
        """
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided either directly or via GROQ_API_KEY environment variable")
        
        self.model = model
        self.client = Groq(api_key=self.api_key)
        
    def _create_prompt(self, features, audio_filename, metadata=None):
        """
        Create a prompt for the LLM based on the extracted features.
        
        Parameters:
        -----------
        features : dict
            Dictionary containing speech fluency features
        audio_filename : str
            Name of the analyzed audio file
        metadata : dict, optional
            Additional metadata about the speaker or recording
            
        Returns:
        --------
        str
            Formatted prompt for the LLM
        """
        # Format the features with clear descriptions
        features_description = {
            "rate_of_fbds_speech_segments": "Rate of speech segments per second",
            "std_dev_fbds_speech_segments_duration": "Standard deviation of speech segment durations (seconds)",
            "rate_of_fbds_silent_segments": "Rate of silent segments per second",
            "speech_ratio": "Proportion of time spent speaking vs. total time",
            "rate_of_pseudo_syllables": "Rate of pseudo-syllables per second (syllable-like units)",
            "std_dev_pseudo_syllable_duration": "Standard deviation of pseudo-syllable durations (seconds)",
            "rate_of_silent_breaks": "Rate of significant pauses per second"
        }
        
        formatted_features = "\n".join([
            f"- {desc}: {features[key]:.4f}" for key, desc in features_description.items()
        ])
        
        # Include metadata if provided
        metadata_text = ""
        if metadata:
            metadata_text = "\nAdditional context:\n"
            for key, value in metadata.items():
                metadata_text += f"- {key}: {value}\n"
        
        # Create the prompt
        prompt = f"""
You are an expert in speech analysis and language assessment. You have been provided with quantitative metrics extracted from an audio recording ({audio_filename}).
Please follow the instructions below:
1. If the speech ratio is less than 0.55 then it is not effictively use of his/her speaking time.
2. if there is high rate of silent segments then it is not good.
3. If rate of pesudo variables is greater than 20 it is not good.

Please analyze these metrics and provide:
1. An assessment of the speaker's fluency (on a scale of 1-10). It is high when you find high speech ratio, low rate of silent segments, low rate of pseudo variables.
2. Strengths and areas for improvement
3. Specific actionable advice to improve fluency
4. A comparative analysis to typical fluency patterns

Speech Fluency Metrics:
{formatted_features}
{metadata_text}

Interpret these metrics in a way that would be helpful for language learners or speech therapists. Structure your response with clear headings and bullet points where appropriate.
"""
        return prompt
        
    def analyze_fluency(self, features, audio_filename, metadata=None):
        """
        Analyze speech fluency features using an LLM.
        
        Parameters:
        -----------
        features : dict
            Dictionary containing speech fluency features
        audio_filename : str
            Name of the analyzed audio file
        metadata : dict, optional
            Additional metadata about the speaker or recording
            
        Returns:
        --------
        dict
            LLM analysis results
        """
        prompt = self._create_prompt(features, audio_filename, metadata)
        
        try:
            # Call the Groq LLM API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert speech analyst specializing in fluency assessment."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.2
            )
            
            # Extract and return the analysis
            analysis = response.choices[0].message.content
            
            return {
                "success": True,
                "analysis": analysis,
                "model_used": self.model
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "model_used": self.model
            }

def analyze_speech_fluency(audio_file_path, api_key=None, metadata=None):
    """
    Main function to analyze speech fluency from an audio file.
    This function is designed to be imported and called from app.py.
    
    Parameters:
    -----------
    audio_file_path : str
        Path to the audio file to analyze
    api_key : str, optional
        Groq API key (if not provided in environment)
    metadata : dict, optional
        Additional context about the speaker or recording
        
    Returns:
    --------
    dict
        Complete analysis results including raw metrics and LLM interpretation
    """
    # Create instance of the speech fluency predictor
    predictor = SpeechFluencyPredictor()
    
    # Extract audio filename
    audio_filename = os.path.basename(audio_file_path)
    
    # Process the audio file
    features, segments = predictor.process(audio_file_path)
    
    # Default metadata if none provided
    if metadata is None:
        metadata = {
            "speaker_type": "Unknown",
            "context": "Unspecified speaking context",
            "recording_quality": "Unknown recording quality"
        }
    
    # Create LLM analyzer and get analysis
    try:
        llm_analyzer = FluencyAnalysisLLM(api_key=api_key)
        llm_result = llm_analyzer.analyze_fluency(features, audio_filename, metadata)
    except Exception as e:
        llm_result = {
            "success": False,
            "error": str(e),
            "model_used": "llama3-70b-8192"
        }
    
    # Create complete results object
    results = {
        "audio_file": audio_filename,
        "features": {k: float(v) for k, v in features.items()},  # Convert numpy values to float for JSON
        "segment_counts": {
            "fbds_segments": len(segments['fbds_segments']),
            "speech_segments": len(segments['fbds_speech_segments']),
            "silent_segments": len(segments['fbds_silent_segments']),
            "pseudo_syllables": len(segments['pseudo_syllables']),
            "silent_breaks": len(segments['silent_breaks'])
        },
        "llm_analysis": llm_result
    }
    
    # Save results to reports directory
    os.makedirs('reports', exist_ok=True)
    report_path = f"reports/{os.path.splitext(audio_filename)[0]}_analysis.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    
    return results

# Example usage - this part won't run when imported from app.py
if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Analyze speech fluency from audio file')
    parser.add_argument('audio_file', help='Path to audio file')
    parser.add_argument('--api_key', help='Groq API key (optional if set in environment)')
    args = parser.parse_args()
    
    # Simple metadata for standalone usage
    metadata = {
        "speaker_type": "Unknown speaker",
        "context": "Command line analysis",
        "recording_quality": "Unknown quality"
    }
    
    # Run analysis
    results = analyze_speech_fluency(args.audio_file, args.api_key, metadata)
    
    # Print summary of results
    print(f"\nAnalysis complete for {results['audio_file']}")
    print("\nSpeech Fluency Features:")
    for feature, value in results['features'].items():
        print(f"  {feature}: {value:.4f}")
    
    print("\nSegmentation Statistics:")
    for segment_type, count in results['segment_counts'].items():
        print(f"  {segment_type}: {count}")
    
    if results['llm_analysis']['success']:
        print("\nLLM Analysis:")
        print(results['llm_analysis']['analysis'])
    else:
        print(f"\nLLM Analysis failed: {results['llm_analysis']['error']}")
    
    print(f"\nFull results saved to reports/{os.path.splitext(results['audio_file'])[0]}_analysis.json")