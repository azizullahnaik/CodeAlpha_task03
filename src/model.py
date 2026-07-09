"""
LSTM Model Architecture for Music Generation
Deep recurrent neural network with multiple LSTM layers, dropout, and dense output
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import numpy as np


class JazzLSTM:
    """LSTM-based music generation model"""
    
    def __init__(self, input_shape, output_size, lstm_units=[256, 256, 128], dropout_rate=0.3):
        self.input_shape = input_shape
        self.output_size = output_size
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = None
        
    def build_model(self):
        """Build the LSTM model architecture"""
        model = Sequential()
        
        # First LSTM layer with return sequences
        model.add(LSTM(
            self.lstm_units[0],
            input_shape=self.input_shape,
            return_sequences=True,
            activation='tanh',
            recurrent_activation='sigmoid',
            recurrent_dropout=0.0
        ))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))
        
        # Second LSTM layer with return sequences
        model.add(LSTM(
            self.lstm_units[1],
            return_sequences=True,
            activation='tanh',
            recurrent_activation='sigmoid',
            recurrent_dropout=0.0
        ))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))
        
        # Third LSTM layer (no return sequences)
        if len(self.lstm_units) > 2:
            model.add(LSTM(
                self.lstm_units[2],
                return_sequences=False,
                activation='tanh',
                recurrent_activation='sigmoid',
                recurrent_dropout=0.0
            ))
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout_rate))
        
        # Dense layers
        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))
        
        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))
        
        # Output layer with softmax for probability distribution
        model.add(Dense(self.output_size, activation='softmax'))
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model with optimizer and loss function"""
        if self.model is None:
            self.build_model()
        
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',  # Use sparse for integer labels
            metrics=['accuracy']
        )
        
        return self.model
    
    def get_model_summary(self):
        """Get model architecture summary"""
        if self.model is None:
            self.build_model()
        return self.model.summary()
    
    def count_parameters(self):
        """Count total and trainable parameters"""
        if self.model is None:
            self.build_model()
        
        total_params = self.model.count_params()
        trainable_params = sum([tf.size(w).numpy() for w in self.model.trainable_weights])
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }


def create_callbacks(checkpoint_path='output/best_model.keras'):
    """Create training callbacks"""
    callbacks = [
        # Save the best model
        ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate when plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    return callbacks


def build_advanced_model(input_shape, output_size, lstm_units=[512, 512, 256], dropout_rate=0.4):
    """Build a more advanced LSTM model with attention-like architecture"""
    
    inputs = Input(shape=input_shape)
    
    # First LSTM layer
    x = LSTM(lstm_units[0], return_sequences=True, activation='tanh')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    # Second LSTM layer
    x = LSTM(lstm_units[1], return_sequences=True, activation='tanh')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    # Third LSTM layer
    x = LSTM(lstm_units[2], return_sequences=False, activation='tanh')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    # Dense layers
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    # Output layer
    outputs = Dense(output_size, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    return model


if __name__ == '__main__':
    # Test model creation
    print("Testing LSTM Model Creation...")
    
    # Sample input shape (sequence_length, features)
    input_shape = (100, 1)
    output_size = 359  # Typical number of unique notes in jazz dataset
    
    # Create model
    jazz_model = JazzLSTM(input_shape, output_size, lstm_units=[256, 256, 128])
    model = jazz_model.build_model()
    model = jazz_model.compile_model(learning_rate=0.001)
    
    # Print summary
    jazz_model.get_model_summary()
    
    # Print parameter count
    params = jazz_model.count_parameters()
    print(f"\nTotal Parameters: {params['total_parameters']:,}")
    print(f"Trainable Parameters: {params['trainable_parameters']:,}")
    
    print("\nModel architecture created successfully!")
