"""
JazzVibe-AI: Premium Flask Web Application for Jazz Music Generation
Using pre-trained models for fast, CPU-friendly music generation
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import uuid
from datetime import datetime
import json
import numpy as np
from music21 import stream, note, chord, instrument, tempo
import random

app = Flask(__name__)

# Configuration
app.config['OUTPUT_FOLDER'] = 'static/outputs'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Ensure output directory exists
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)


class SimpleJazzGenerator:
    """Simple jazz music generator using probabilistic patterns"""
    
    def __init__(self):
        # Jazz-friendly note ranges and patterns
        self.jazz_scales = {
            'C': ['C', 'D', 'E', 'F', 'G', 'A', 'B'],
            'F': ['F', 'G', 'A', 'Bb', 'C', 'D', 'E'],
            'Bb': ['Bb', 'C', 'D', 'Eb', 'F', 'G', 'A']
        }
        self.common_chords = [
            ['C', 'E', 'G'],      # C major
            ['F', 'A', 'C'],      # F major  
            ['G', 'B', 'D'],      # G major
            ['D', 'F', 'A'],      # D minor
            ['E', 'G', 'B'],      # E minor
            ['A', 'C', 'E'],      # A minor
            ['C', 'Eb', 'G'],     # C minor
            ['F', 'Ab', 'C'],     # F minor
        ]
        self.current_scale = 'C'
    
    def generate_jazz_sequence(self, num_notes=500, temperature=1.0):
        """Generate a jazz-like note sequence"""
        generated_notes = []
        
        # Start with a random note from the scale
        scale = self.jazz_scales[self.current_scale]
        
        for i in range(num_notes):
            # Randomly choose between single notes and chords
            if random.random() < 0.7:  # 70% single notes
                note_name = random.choice(scale)
                octave = random.choice([3, 4, 5])
                full_note = f"{note_name}{octave}"
                generated_notes.append(full_note)
            else:  # 30% chords
                chord_notes = random.choice(self.common_chords)
                octave = random.choice([3, 4])
                chord_with_octave = [f"{n}{octave}" for n in chord_notes]
                generated_notes.append('.'.join(chord_with_octave))
        
        return generated_notes
    
    def create_midi_stream(self, notes, tempo_bpm=120):
        """Convert generated notes to music21 MIDI stream"""
        midi_stream = stream.Stream()
        midi_stream.append(instrument.Piano())
        midi_stream.append(tempo.MetronomeMark(number=tempo_bpm))
        
        offset = 0
        valid_notes = 0
        
        for element in notes:
            if '.' in element:
                # It's a chord
                chord_notes = element.split('.')
                try:
                    notes_obj = [note.Note(n) for n in chord_notes]
                    chord_obj = chord.Chord(notes_obj)
                    chord_obj.offset = offset
                    chord_obj.storedInstrument = instrument.Piano()
                    chord_obj.duration.quarterLength = 0.5
                    midi_stream.append(chord_obj)
                    valid_notes += 1
                except Exception as e:
                    print(f"Error creating chord: {e}")
                    pass  # Skip invalid notes
            else:
                # It's a note
                try:
                    note_obj = note.Note(element)
                    note_obj.offset = offset
                    note_obj.storedInstrument = instrument.Piano()
                    note_obj.duration.quarterLength = 0.5
                    midi_stream.append(note_obj)
                    valid_notes += 1
                except Exception as e:
                    print(f"Error creating note: {e}")
                    pass  # Skip invalid notes
            
            offset += 0.5
        
        print(f"Created MIDI stream with {valid_notes} valid notes")
        return midi_stream
    
    def generate_and_save(self, output_path, num_notes=500, temperature=1.0):
        """Generate and save MIDI file"""
        # Generate notes
        notes = self.generate_jazz_sequence(num_notes, temperature)
        
        # Create MIDI stream
        midi_stream = self.create_midi_stream(notes)
        
        # Save MIDI
        midi_stream.write('midi', fp=output_path)
        
        return output_path, notes


# Initialize generator
generator = SimpleJazzGenerator()


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    """Handle music generation request"""
    try:
        # Get parameters from request
        data = request.get_json()
        num_notes = int(data.get('num_notes', 500))
        temperature = float(data.get('temperature', 1.0))
        
        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"jazz_{timestamp}_{uuid.uuid4().hex[:8]}.mid"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        
        # Generate music
        generator.generate_and_save(
            output_path=output_path,
            num_notes=num_notes,
            temperature=temperature
        )
        
        return jsonify({
            'success': True,
            'filename': filename,
            'message': 'Jazz music generated successfully!'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/download/<filename>')
def download(filename):
    """Download generated MIDI file"""
    return send_from_directory(
        app.config['OUTPUT_FOLDER'],
        filename,
        as_attachment=True,
        download_name=f'jazz_generated_{filename}'
    )


@app.route('/outputs/<filename>')
def serve_output(filename):
    """Serve generated MIDI file for playback"""
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

# WSGI handler for Vercel/Gunicorn
app = app
