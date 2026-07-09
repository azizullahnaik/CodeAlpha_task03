"""
Training Pipeline for Jazz LSTM Model
Handles model training with callbacks, checkpoints, and progress tracking
"""

import os
import sys
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
import json
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import JazzLSTM, create_callbacks


class JazzTrainer:
    """Handles training of the Jazz LSTM model"""
    
    def __init__(self, data_dir='output', model_dir='output', epochs=100, batch_size=64):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.history = None
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
    def load_preprocessed_data(self):
        """Load preprocessed data from disk"""
        print("Loading preprocessed data...")
        
        with open(os.path.join(self.data_dir, 'notes.pkl'), 'rb') as f:
            notes = pickle.load(f)
        
        with open(os.path.join(self.data_dir, 'note_to_int.pkl'), 'rb') as f:
            note_to_int = pickle.load(f)
        
        with open(os.path.join(self.data_dir, 'int_to_note.pkl'), 'rb') as f:
            int_to_note = pickle.load(f)
        
        print(f"Loaded {len(notes)} notes with {len(note_to_int)} unique notes")
        
        # Try to load pre-computed sequences if available
        network_input = None
        network_output = None
        
        if os.path.exists(os.path.join(self.data_dir, 'network_input.npy')):
            network_input = np.load(os.path.join(self.data_dir, 'network_input.npy'))
            print(f"Loaded pre-computed network_input: {network_input.shape}")
        
        if os.path.exists(os.path.join(self.data_dir, 'network_output.npy')):
            network_output = np.load(os.path.join(self.data_dir, 'network_output.npy'))
            print(f"Loaded pre-computed network_output: {network_output.shape}")
        
        return notes, note_to_int, int_to_note, network_input, network_output
    
    def prepare_training_data(self, network_input_raw, network_output_raw, note_to_int, sequence_length=100):
        """Prepare training data from pre-computed sequences (memory-efficient)"""
        print("Preparing training sequences...")
        
        n_patterns = len(network_input_raw)
        print(f"Processing {n_patterns} training sequences")
        
        # Reshape input for LSTM [samples, time steps, features]
        network_input = network_input_raw.reshape(n_patterns, sequence_length, 1)
        
        # Normalize input to float32 for memory efficiency
        network_input = network_input.astype(np.float32) / float(len(note_to_int))
        
        # Output remains as integer labels for sparse categorical crossentropy
        network_output = network_output_raw.astype(np.int32)
        
        print(f"Memory usage - Input: {network_input.nbytes / 1024 / 1024:.2f} MB")
        print(f"Memory usage - Output: {network_output.nbytes / 1024 / 1024:.2f} MB")
        
        return network_input, network_output
    
    def build_model(self, input_shape, output_size):
        """Build and compile the model"""
        print("Building LSTM model...")
        
        self.model = JazzLSTM(
            input_shape=input_shape,
            output_size=output_size,
            lstm_units=[256, 256, 128],
            dropout_rate=0.3
        )
        
        model = self.model.build_model()
        model = self.model.compile_model(learning_rate=0.001)
        
        # Print model summary
        print("\n" + "="*60)
        print("MODEL ARCHITECTURE")
        print("="*60)
        self.model.get_model_summary()
        
        params = self.model.count_parameters()
        print(f"\nTotal Parameters: {params['total_parameters']:,}")
        print(f"Trainable Parameters: {params['trainable_parameters']:,}")
        print("="*60 + "\n")
        
        return model
    
    def get_callbacks(self):
        """Create training callbacks"""
        checkpoint_path = os.path.join(self.model_dir, 'best_model.keras')
        
        callbacks = [
            # Save the best model
            ModelCheckpoint(
                checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=1
            ),
            
            # Early stopping
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            ),
            
            # TensorBoard logging
            TensorBoard(
                log_dir=os.path.join(self.model_dir, 'logs'),
                histogram_freq=1
            )
        ]
        
        return callbacks
    
    def train(self, network_input, network_output, validation_split=0.2):
        """Train the model"""
        print(f"\nStarting training for {self.epochs} epochs...")
        print(f"Batch size: {self.batch_size}")
        print(f"Validation split: {validation_split}")
        print("-" * 60)
        
        callbacks = self.get_callbacks()
        
        self.history = self.model.model.fit(
            network_input,
            network_output,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\nTraining completed!")
        return self.history
    
    def save_training_history(self):
        """Save training history to JSON"""
        if self.history is None:
            print("No training history to save")
            return
        
        history_dict = {
            'loss': [float(x) for x in self.history.history['loss']],
            'accuracy': [float(x) for x in self.history.history['accuracy']],
            'val_loss': [float(x) for x in self.history.history['val_loss']],
            'val_accuracy': [float(x) for x in self.history.history['val_accuracy']],
            'epochs': len(self.history.history['loss'])
        }
        
        history_path = os.path.join(self.model_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        print(f"Training history saved to {history_path}")
    
    def save_final_model(self):
        """Save the final trained model"""
        final_model_path = os.path.join(self.model_dir, 'final_model.keras')
        self.model.model.save(final_model_path)
        print(f"Final model saved to {final_model_path}")


def main():
    """Main training pipeline"""
    print("="*60)
    print("JAZZ LSTM MODEL TRAINING")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60 + "\n")
    
    # Initialize trainer (fast local testing settings)
    trainer = JazzTrainer(
        data_dir='output',
        model_dir='output',
        epochs=5,  # Reduced for fast testing
        batch_size=512  # Increased for faster CPU processing
    )
    
    # Load preprocessed data (including pre-computed sequences if available)
    notes, note_to_int, int_to_note, network_input_raw, network_output_raw = trainer.load_preprocessed_data()
    
    # Check if pre-computed sequences exist
    if network_input_raw is None or network_output_raw is None:
        print("ERROR: Pre-computed sequences not found. Please run preprocess.py first.")
        return
    
    # Prepare training data from pre-computed sequences
    sequence_length = 100
    network_input, network_output = trainer.prepare_training_data(
        network_input_raw, network_output_raw, note_to_int, sequence_length
    )
    
    # Slice to first 20,000 samples for fast local testing
    max_samples = 20000
    if len(network_input) > max_samples:
        print(f"Slicing data to first {max_samples} samples for fast training")
        network_input = network_input[:max_samples]
        network_output = network_output[:max_samples]
        print(f"Training data shape: {network_input.shape}")
    
    # Build model
    input_shape = (sequence_length, 1)
    output_size = len(note_to_int)
    trainer.build_model(input_shape, output_size)
    
    # Train model
    trainer.train(network_input, network_output, validation_split=0.2)
    
    # Save training history and final model
    trainer.save_training_history()
    trainer.save_final_model()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)


if __name__ == '__main__':
    main()
