"""
MIDI Data Preprocessing Script
Parses MIDI files, extracts notes/chords, and creates sequences for LSTM training
"""

import os
import pickle
import numpy as np
import pandas as pd
from music21 import converter, instrument, note, chord
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class MIDIPreprocessor:
    """Handles MIDI file preprocessing for music generation"""
    
    def __init__(self, data_dir, sequence_length=100):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.notes = []
        self.note_to_int = {}
        self.int_to_note = {}
        
    def get_midi_files(self):
        """Get all MIDI files from the data directory"""
        midi_files = []
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.mid') or file.endswith('.midi'):
                    midi_files.append(os.path.join(root, file))
        return midi_files
    
    def parse_midi_file(self, file_path):
        """Parse a single MIDI file and extract notes/chords"""
        try:
            midi = converter.parse(file_path)
            notes_to_parse = []
            
            # Flatten the MIDI file to get all notes and chords
            for element in midi.flat:
                if isinstance(element, note.Note):
                    notes_to_parse.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes_to_parse.append('.'.join(str(n) for n in element.normalOrder))
            
            return notes_to_parse
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return []
    
    def extract_notes(self, max_files=150):
        """Extract notes from MIDI files (limited to prevent memory issues)"""
        midi_files = self.get_midi_files()
        
        # Limit to max_files to prevent memory issues
        if max_files and len(midi_files) > max_files:
            import random
            random.seed(42)  # For reproducibility
            midi_files = random.sample(midi_files, max_files)
            print(f"Sampling {max_files} files from {len(midi_files)} total files")
        elif max_files:
            midi_files = midi_files[:max_files]
        
        print(f"Processing {len(midi_files)} MIDI files")
        print("Extracting notes from MIDI files...")
        
        for file in tqdm(midi_files):
            notes = self.parse_midi_file(file)
            self.notes.extend(notes)
        
        print(f"Total notes extracted: {len(self.notes)}")
        return self.notes
    
    def create_note_mapping(self):
        """Create mapping between notes and integers"""
        unique_notes = sorted(set(self.notes))
        self.note_to_int = {note: idx for idx, note in enumerate(unique_notes)}
        self.int_to_note = {idx: note for note, idx in self.note_to_int.items()}
        
        print(f"Unique notes: {len(unique_notes)}")
        return self.note_to_int, self.int_to_note
    
    def create_sequences(self):
        """Create input/output sequences for training (memory-efficient)"""
        if not self.note_to_int:
            self.create_note_mapping()
        
        network_input = []
        network_output = []
        
        print(f"Creating sequences of length {self.sequence_length}...")
        
        for i in range(len(self.notes) - self.sequence_length):
            sequence_in = self.notes[i:i + self.sequence_length]
            sequence_out = self.notes[i + self.sequence_length]
            
            network_input.append([self.note_to_int[note] for note in sequence_in])
            network_output.append(self.note_to_int[sequence_out])
        
        n_patterns = len(network_input)
        print(f"Total sequences created: {n_patterns}")
        
        # Convert to numpy arrays with efficient dtypes (no one-hot encoding)
        # Use uint16 for input to save memory (assuming < 65535 unique notes)
        network_input = np.array(network_input, dtype=np.uint16)
        network_output = np.array(network_output, dtype=np.uint16)
        
        print(f"Memory usage - Input: {network_input.nbytes / 1024 / 1024:.2f} MB")
        print(f"Memory usage - Output: {network_output.nbytes / 1024 / 1024:.2f} MB")
        
        return network_input, network_output
    
    def save_preprocessed_data(self, output_dir, network_input=None, network_output=None):
        """Save preprocessed data to disk"""
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, 'notes.pkl'), 'wb') as f:
            pickle.dump(self.notes, f)
        
        with open(os.path.join(output_dir, 'note_to_int.pkl'), 'wb') as f:
            pickle.dump(self.note_to_int, f)
        
        with open(os.path.join(output_dir, 'int_to_note.pkl'), 'wb') as f:
            pickle.dump(self.int_to_note, f)
        
        # Save network input and output if provided
        if network_input is not None:
            np.save(os.path.join(output_dir, 'network_input.npy'), network_input)
            print(f"Saved network_input to {output_dir}/network_input.npy")
        
        if network_output is not None:
            np.save(os.path.join(output_dir, 'network_output.npy'), network_output)
            print(f"Saved network_output to {output_dir}/network_output.npy")
        
        print(f"Preprocessed data saved to {output_dir}")
    
    def load_preprocessed_data(self, input_dir):
        """Load preprocessed data from disk"""
        with open(os.path.join(input_dir, 'notes.pkl'), 'rb') as f:
            self.notes = pickle.load(f)
        
        with open(os.path.join(input_dir, 'note_to_int.pkl'), 'rb') as f:
            self.note_to_int = pickle.load(f)
        
        with open(os.path.join(input_dir, 'int_to_note.pkl'), 'rb') as f:
            self.int_to_note = pickle.load(f)
        
        print(f"Preprocessed data loaded from {input_dir}")
        return self.notes, self.note_to_int, self.int_to_note
    
    def get_statistics(self):
        """Get dataset statistics"""
        unique_notes = len(set(self.notes))
        total_notes = len(self.notes)
        avg_sequence_length = np.mean([len(self.notes[i:i+self.sequence_length]) 
                                      for i in range(len(self.notes)-self.sequence_length)])
        
        stats = {
            'total_midi_files': len(self.get_midi_files()),
            'total_notes': total_notes,
            'unique_notes': unique_notes,
            'sequence_length': self.sequence_length,
            'avg_sequence_length': avg_sequence_length
        }
        return stats


def main():
    """Main preprocessing pipeline"""
    # Initialize preprocessor
    data_dir = 'data/Jazz Midi'
    preprocessor = MIDIPreprocessor(data_dir, sequence_length=100)
    
    # Extract notes from MIDI files (limited to 150 files for memory efficiency)
    preprocessor.extract_notes(max_files=150)
    
    # Create note mapping
    preprocessor.create_note_mapping()
    
    # Create sequences (memory-efficient, no one-hot encoding)
    network_input, network_output = preprocessor.create_sequences()
    
    # Save preprocessed data including network input/output
    preprocessor.save_preprocessed_data('output', network_input, network_output)
    
    # Print statistics
    stats = preprocessor.get_statistics()
    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    for key, value in stats.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    print("="*50)
    
    return network_input, network_output


if __name__ == '__main__':
    network_input, network_output = main()
