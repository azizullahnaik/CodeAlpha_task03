"""
Music Generation Inference Script
Generates jazz music using trained LSTM model with temperature-based sampling
"""

import os
import sys
import pickle
import numpy as np
import tensorflow as tf
from music21 import stream, note, chord, instrument, tempo
from tqdm import tqdm
import random


class JazzGenerator:
    """Handles music generation using trained LSTM model"""
    
    def __init__(self, model_path, data_dir='output'):
        self.model_path = model_path
        self.data_dir = data_dir
        self.model = None
        self.note_to_int = {}
        self.int_to_note = {}
        self.notes = []
        
        # Load model and data
        self.load_model()
        self.load_preprocessed_data()
    
    def load_model(self):
        """Load the trained model"""
        print(f"Loading model from {self.model_path}...")
        self.model = tf.keras.models.load_model(self.model_path)
        print("Model loaded successfully!")
    
    def load_preprocessed_data(self):
        """Load preprocessed data for generation"""
        print("Loading preprocessed data...")
        
        with open(os.path.join(self.data_dir, 'notes.pkl'), 'rb') as f:
            self.notes = pickle.load(f)
        
        with open(os.path.join(self.data_dir, 'note_to_int.pkl'), 'rb') as f:
            self.note_to_int = pickle.load(f)
        
        with open(os.path.join(self.data_dir, 'int_to_note.pkl'), 'rb') as f:
            self.int_to_note = pickle.load(f)
        
        print(f"Loaded {len(self.notes)} notes with {len(self.note_to_int)} unique notes")
    
    def sample_with_temperature(self, predictions, temperature=1.0):
        """Sample from probability distribution with temperature"""
        predictions = np.asarray(predictions).astype('float64')
        predictions = np.log(predictions) / temperature
        exp_predictions = np.exp(predictions)
        predictions = exp_predictions / np.sum(exp_predictions)
        probas = np.random.multinomial(1, predictions, 1)
        return np.argmax(probas)
    
    def generate_music(self, sequence_length=100, num_notes=500, temperature=1.0, seed=None):
        """Generate jazz music sequence"""
        print(f"\nGenerating music with temperature={temperature}...")
        print(f"Sequence length: {sequence_length}")
        print(f"Number of notes to generate: {num_notes}")
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Get a random starting sequence
        start_index = np.random.randint(0, len(self.notes) - sequence_length - 1)
        pattern = self.notes[start_index:start_index + sequence_length]
        
        print(f"Starting sequence: {pattern[:5]}...{pattern[-5:]}")
        
        generated_notes = []
        
        # Generate notes
        for i in tqdm(range(num_notes), desc="Generating notes"):
            # Convert pattern to integers
            x = [self.note_to_int[note] for note in pattern]
            
            # Reshape and normalize
            x = np.reshape(x, (1, len(x), 1))
            x = x / float(len(self.note_to_int))
            
            # Predict next note
            prediction = self.model.predict(x, verbose=0)[0]
            
            # Sample with temperature
            index = self.sample_with_temperature(prediction, temperature)
            
            # Convert back to note
            result = self.int_to_note[index]
            generated_notes.append(result)
            
            # Update pattern
            pattern.append(result)
            pattern = pattern[1:]
        
        print(f"Generated {len(generated_notes)} notes")
        return generated_notes
    
    def create_midi_stream(self, notes, tempo_bpm=120):
        """Convert generated notes to music21 MIDI stream"""
        print("Creating MIDI stream...")
        
        # Create a music stream
        midi_stream = stream.Stream()
        
        # Add piano instrument
        midi_stream.append(instrument.Piano())
        
        # Add tempo
        midi_stream.append(tempo.MetronomeMark(number=tempo_bpm))
        
        # Parse notes and chords
        offset = 0
        for element in notes:
            if '.' in element:
                # It's a chord
                chord_notes = element.split('.')
                notes_obj = [note.Note(int(n)) for n in chord_notes]
                chord_obj = chord.Chord(notes_obj)
                chord_obj.offset = offset
                chord_obj.storedInstrument = instrument.Piano()
                midi_stream.append(chord_obj)
            else:
                # It's a note
                note_obj = note.Note(int(element))
                note_obj.offset = offset
                note_obj.storedInstrument = instrument.Piano()
                midi_stream.append(note_obj)
            
            # Increase offset
            offset += 0.5
        
        print("MIDI stream created!")
        return midi_stream
    
    def save_midi(self, midi_stream, output_path):
        """Save MIDI stream to file"""
        print(f"Saving MIDI to {output_path}...")
        midi_stream.write('midi', fp=output_path)
        print(f"MIDI saved successfully!")
    
    def generate_and_save(self, output_path='output/generated_jazz.mid', 
                         sequence_length=100, num_notes=500, temperature=1.0, seed=None):
        """Complete generation pipeline: generate and save MIDI"""
        # Generate notes
        generated_notes = self.generate_music(
            sequence_length=sequence_length,
            num_notes=num_notes,
            temperature=temperature,
            seed=seed
        )
        
        # Create MIDI stream
        midi_stream = self.create_midi_stream(generated_notes)
        
        # Save MIDI
        self.save_midi(midi_stream, output_path)
        
        return output_path, generated_notes


def generate_multiple_versions(model_path, output_dir='output', num_versions=5, 
                             temperatures=[0.7, 1.0, 1.3]):
    """Generate multiple versions with different temperatures"""
    print(f"Generating {num_versions} versions with different temperatures...")
    
    generator = JazzGenerator(model_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i, temp in enumerate(temperatures):
        output_path = os.path.join(output_dir, f'generated_jazz_temp{temp:.1f}_v{i+1}.mid')
        print(f"\nGenerating version {i+1} with temperature {temp}...")
        
        generator.generate_and_save(
            output_path=output_path,
            sequence_length=100,
            num_notes=500,
            temperature=temp
        )
    
    print(f"\nAll versions saved to {output_dir}")


def main():
    """Main generation pipeline"""
    print("="*60)
    print("JAZZ MUSIC GENERATION")
    print("="*60)
    
    # Check if model exists
    model_path = 'output/best_model.keras'
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please train the model first using train.py")
        return
    
    # Initialize generator
    generator = JazzGenerator(model_path)
    
    # Generate music with different parameters
    print("\n" + "-"*60)
    print("Generating with default parameters (temperature=1.0)")
    print("-"*60)
    output_path, notes = generator.generate_and_save(
        output_path='output/generated_jazz.mid',
        sequence_length=100,
        num_notes=500,
        temperature=1.0
    )
    
    print("\n" + "-"*60)
    print("Generating with low temperature (temperature=0.7) - more conservative")
    print("-"*60)
    output_path_low, notes_low = generator.generate_and_save(
        output_path='output/generated_jazz_low_temp.mid',
        sequence_length=100,
        num_notes=500,
        temperature=0.7
    )
    
    print("\n" + "-"*60)
    print("Generating with high temperature (temperature=1.3) - more creative")
    print("-"*60)
    output_path_high, notes_high = generator.generate_and_save(
        output_path='output/generated_jazz_high_temp.mid',
        sequence_length=100,
        num_notes=500,
        temperature=1.3
    )
    
    print("\n" + "="*60)
    print("GENERATION COMPLETED")
    print("="*60)
    print(f"Generated files:")
    print(f"  - {output_path}")
    print(f"  - {output_path_low}")
    print(f"  - {output_path_high}")
    print("="*60)


if __name__ == '__main__':
    main()
