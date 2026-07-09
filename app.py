"""
JazzVibe-AI: Premium Flask Web Application for Jazz Music Generation
Using pre-trained models for fast, CPU-friendly music generation
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import uuid
from datetime import datetime
import json

app = Flask(__name__)

# Configuration
app.config['OUTPUT_FOLDER'] = 'static/outputs'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Ensure output directory exists
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
import streamlit as st
import tensorflow as tf
from music21 import converter, note, chord, instrument
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import base64
from io import BytesIO

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Page configuration
st.set_page_config(
    page_title="JazzVibe-AI",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def load_css():
    with open('assets/style.css', 'r') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

# Session state initialization
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'preprocessed' not in st.session_state:
    st.session_state.preprocessed = False
if 'generated_files' not in st.session_state:
    st.session_state.generated_files = []

# Helper functions
def get_midi_files_count():
    """Count MIDI files in data directory"""
    count = 0
    if os.path.exists('data/Jazz Midi'):
        for root, dirs, files in os.walk('data/Jazz Midi'):
            count += len([f for f in files if f.endswith('.mid')])
    return count

def load_preprocessed_stats():
    """Load preprocessed data statistics"""
    if os.path.exists('output/notes.pkl'):
        with open('output/notes.pkl', 'rb') as f:
            notes = pickle.load(f)
        with open('output/note_to_int.pkl', 'rb') as f:
            note_to_int = pickle.load(f)
        
        return {
            'total_notes': len(notes),
            'unique_notes': len(note_to_int),
            'midi_files': get_midi_files_count()
        }
    return None

def create_note_distribution_chart(notes_sample):
    """Create a beautiful note distribution chart"""
    if not notes_sample:
        return None
    
    note_counts = pd.Series(notes_sample).value_counts().head(20)
    
    fig = go.Figure(data=[
        go.Bar(
            x=note_counts.index,
            y=note_counts.values,
            marker=dict(
                color=note_counts.values,
                colorscale='Viridis',
                showscale=True
            ),
            text=note_counts.values,
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title="Top 20 Most Frequent Notes",
        xaxis_title="Notes",
        yaxis_title="Frequency",
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8'),
        title_font=dict(color='#fff', size=18),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            tickfont=dict(color='#94a3b8')
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            tickfont=dict(color='#94a3b8')
        )
    )
    
    return fig

def create_training_progress_chart():
    """Create training progress visualization"""
    if not os.path.exists('output/training_history.json'):
        return None
    
    import json
    with open('output/training_history.json', 'r') as f:
        history = json.load(f)
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Model Loss', 'Model Accuracy'),
        vertical_spacing=0.15
    )
    
    # Loss plot
    fig.add_trace(
        go.Scatter(
            y=history['loss'],
            name='Training Loss',
            line=dict(color='#8b5cf6', width=2)
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            y=history['val_loss'],
            name='Validation Loss',
            line=dict(color='#6366f1', width=2, dash='dash')
        ),
        row=1, col=1
    )
    
    # Accuracy plot
    fig.add_trace(
        go.Scatter(
            y=history['accuracy'],
            name='Training Accuracy',
            line=dict(color='#8b5cf6', width=2),
            showlegend=False
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            y=history['val_accuracy'],
            name='Validation Accuracy',
            line=dict(color='#6366f1', width=2, dash='dash'),
            showlegend=False
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=600,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8'),
        title_font=dict(color='#fff', size=18),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )
    
    fig.update_xaxes(title_text="Epoch", gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
    
    return fig

def midi_to_audio_html(midi_path):
    """Convert MIDI to playable audio in browser"""
    try:
        # Read MIDI file
        with open(midi_path, 'rb') as f:
            midi_bytes = f.read()
        
        # Create base64 encoded audio
        b64 = base64.b64encode(midi_bytes).decode()
        
        # Create HTML for download
        html = f"""
        <div class="audio-player">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div style="flex: 1;">
                    <p style="color: #fff; margin: 0; font-weight: 500;">🎵 Generated Jazz MIDI</p>
                    <p style="color: #94a3b8; margin: 0.25rem 0 0 0; font-size: 0.9rem;">Ready to download</p>
                </div>
                <a href="data:audio/midi;base64,{b64}" 
                   download="generated_jazz.mid"
                   class="download-button">
                    ⬇️ Download MIDI
                </a>
            </div>
        </div>
        """
        return html
    except Exception as e:
        return f'<div class="error-message">Error loading audio: {str(e)}</div>'

# Main UI
def main():
    # Header
    st.markdown("""
    <div class="main-container">
        <h1 class="header-title">🎵 JazzVibe-AI</h1>
        <p class="header-subtitle">Deep Learning Jazz Music Generation with LSTM Networks</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        # Generation parameters
        st.markdown("#### Generation Parameters")
        sequence_length = st.slider(
            "Sequence Length",
            min_value=50,
            max_value=200,
            value=100,
            step=10
        )
        
        num_notes = st.slider(
            "Number of Notes to Generate",
            min_value=100,
            max_value=1000,
            value=500,
            step=50
        )
        
        temperature = st.slider(
            "Temperature (Creativity)",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="Lower values = more conservative, Higher values = more creative"
        )
        
        st.markdown("---")
        
        # Model selection
        st.markdown("#### Model Selection")
        model_path = st.selectbox(
            "Select Model",
            ["output/best_model.keras", "output/final_model.keras"],
            index=0 if os.path.exists("output/best_model.keras") else 1
        )
        
        st.markdown("---")
        
        # System info
        st.markdown("#### System Status")
        st.info(f"📁 MIDI Files: {get_midi_files_count()}")
        
        if os.path.exists('output/notes.pkl'):
            st.success("✅ Preprocessed data available")
        else:
            st.warning("⚠️ Preprocessing required")
        
        if os.path.exists(model_path):
            st.success("✅ Model available")
        else:
            st.warning("⚠️ Training required")

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset", "🧠 Training", "🎹 Generate", "📈 Analytics"])
    
    with tab1:
        st.markdown('<h2 style="color: white; margin-bottom: 1.5rem;">Dataset Overview</h2>', unsafe_allow_html=True)
        
        # Dataset statistics
        stats = load_preprocessed_stats()
        
        if stats:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value">{stats['midi_files']:,}</div>
                    <div class="stat-label">MIDI Files</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value">{stats['total_notes']:,}</div>
                    <div class="stat-label">Total Notes</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value">{stats['unique_notes']:,}</div>
                    <div class="stat-label">Unique Notes</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            
            # Note distribution chart
            if os.path.exists('output/notes.pkl'):
                with open('output/notes.pkl', 'rb') as f:
                    notes = pickle.load(f)
                
                notes_sample = notes[:10000]  # Sample for performance
                fig = create_note_distribution_chart(notes_sample)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("""
            <div class="glass-card">
                <h3 style="color: white;">Dataset Not Preprocessed</h3>
                <p style="color: #94a3b8; margin-top: 1rem;">
                    Please preprocess the dataset first to view statistics and charts.
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<h2 style="color: white; margin-bottom: 1.5rem;">Model Training</h2>', unsafe_allow_html=True)
        
        # Training controls
        col1, col2 = st.columns(2)
        
        with col1:
            epochs = st.number_input("Epochs", min_value=1, max_value=500, value=100)
            batch_size = st.selectbox("Batch Size", [32, 64, 128, 256], index=1)
        
        with col2:
            learning_rate = st.selectbox(
                "Learning Rate",
                [0.0001, 0.001, 0.01],
                index=1,
                format_func=lambda x: f"{x:.4f}"
            )
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Training button
        if st.button("🚀 Start Training", type="primary", use_container_width=True):
            if not os.path.exists('output/notes.pkl'):
                st.error("Please preprocess the dataset first!")
            else:
                with st.spinner("Training in progress... This may take a while."):
                    # Import training module
                    from src.train import JazzTrainer
                    
                    trainer = JazzTrainer(
                        data_dir='output',
                        model_dir='output',
                        epochs=epochs,
                        batch_size=batch_size
                    )
                    
                    # Load and prepare data
                    notes, note_to_int, int_to_note = trainer.load_preprocessed_data()
                    network_input, network_output = trainer.prepare_training_data(
                        notes, note_to_int, sequence_length=100
                    )
                    
                    # Build and train model
                    input_shape = (100, 1)
                    output_size = len(note_to_int)
                    trainer.build_model(input_shape, output_size)
                    trainer.train(network_input, network_output, validation_split=0.2)
                    trainer.save_training_history()
                    trainer.save_final_model()
                    
                    st.session_state.model_loaded = True
                    st.success("Training completed successfully!")
        
        # Training progress chart
        if os.path.exists('output/training_history.json'):
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            fig = create_training_progress_chart()
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown('<h2 style="color: white; margin-bottom: 1.5rem;">Generate Jazz Music</h2>', unsafe_allow_html=True)
        
        # Generation controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            gen_sequence = st.slider(
                "Input Sequence Length",
                min_value=50,
                max_value=200,
                value=100
            )
        
        with col2:
            gen_notes = st.slider(
                "Notes to Generate",
                min_value=100,
                max_value=1000,
                value=500
            )
        
        with col3:
            gen_temp = st.slider(
                "Temperature",
                min_value=0.5,
                max_value=2.0,
                value=1.0,
                step=0.1
            )
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Generate button
        if st.button("🎹 Generate Jazz", type="primary", use_container_width=True):
            if not os.path.exists(model_path):
                st.error("Please train the model first!")
            elif not os.path.exists('output/notes.pkl'):
                st.error("Please preprocess the dataset first!")
            else:
                # Show loading animation
                loading_placeholder = st.empty()
                loading_placeholder.markdown("""
                <div class="loading-container">
                    <div class="loading-spinner"></div>
                    <div class="loading-text">Generating beautiful jazz music...</div>
                    <div class="progress-container">
                        <div class="progress-bar" style="width: 0%"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Simulate progress
                for i in range(0, 101, 10):
                    time.sleep(0.3)
                    loading_placeholder.markdown(f"""
                    <div class="loading-container">
                        <div class="loading-spinner"></div>
                        <div class="loading-text">Generating beautiful jazz music... {i}%</div>
                        <div class="progress-container">
                            <div class="progress-bar" style="width: {i}%"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Actual generation
                from src.generate import JazzGenerator
                
                generator = JazzGenerator(model_path)
                output_path = 'output/generated_jazz_web.mid'
                
                generator.generate_and_save(
                    output_path=output_path,
                    sequence_length=gen_sequence,
                    num_notes=gen_notes,
                    temperature=gen_temp
                )
                
                # Clear loading
                loading_placeholder.empty()
                
                # Show success
                st.markdown("""
                <div class="success-message">
                    ✨ Jazz music generated successfully!
                </div>
                """, unsafe_allow_html=True)
                
                # Audio player
                audio_html = midi_to_audio_html(output_path)
                st.markdown(audio_html, unsafe_allow_html=True)
                
                # Add to session state
                st.session_state.generated_files.append(output_path)
        
        # Previously generated files
        if st.session_state.generated_files:
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown('<h3 style="color: white; margin-bottom: 1rem;">Previously Generated</h3>', unsafe_allow_html=True)
            
            for file_path in st.session_state.generated_files[-5:]:  # Show last 5
                if os.path.exists(file_path):
                    audio_html = midi_to_audio_html(file_path)
                    st.markdown(audio_html, unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<h2 style="color: white; margin-bottom: 1.5rem;">Analytics & Insights</h2>', unsafe_allow_html=True)
        
        # Model architecture info
        if os.path.exists(model_path):
            try:
                model = tf.keras.models.load_model(model_path)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    <div class="glass-card">
                        <h3 style="color: white; margin-bottom: 1rem;">Model Architecture</h3>
                        <ul style="color: #94a3b8; line-height: 1.8;">
                            <li><strong>Type:</strong> LSTM Recurrent Neural Network</li>
                            <li><strong>Layers:</strong> 3 LSTM + 2 Dense</li>
                            <li><strong>Activation:</strong> Tanh (LSTM), ReLU (Dense)</li>
                            <li><strong>Output:</strong> Softmax</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    total_params = model.count_params()
                    st.markdown(f"""
                    <div class="glass-card">
                        <h3 style="color: white; margin-bottom: 1rem;">Model Parameters</h3>
                        <div class="stat-value">{total_params:,}</div>
                        <div class="stat-label">Total Parameters</div>
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
        else:
            st.info("Train a model to view analytics")
    
    # Footer
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="footer">
        <p>Built with ❤️ by Azizullah Naik using TensorFlow, Streamlit & Music21</p>
        <p>© 2024 JazzVibe-AI | Deep Learning Music Generation</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
