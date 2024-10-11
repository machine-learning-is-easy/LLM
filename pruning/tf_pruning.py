import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot

# Example model
def create_model():
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(28, 28)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    return model

model = create_model()

# Apply pruning to the entire model
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0, final_sparsity=0.5, begin_step=0, end_step=2000)
}

# Use the pruning wrapper
pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)

pruned_model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])

# Define the pruning callback to update pruning during training
callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
    # Optionally, use pruning summaries
    tfmot.sparsity.keras.PruningSummaries(log_dir='./logs')
]

# Strip pruning to create a smaller, more efficient model for deployment
final_model = tfmot.sparsity.keras.strip_pruning(pruned_model)

final_model.save('pruned_model.h5')