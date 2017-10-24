LEARNING_RATE = 0.01
REGULARIZATION_STRENGTH = 0.1
TRAINING_EPOCHS = 1500
DATA_SIZE = 150
DISPLAY_STEP = 100

INPUT_SIZE = 4
HIDDEN_SIZE_1 = 500
HIDDEN_SIZE_2 = 300
OUTPUT_SIZE = 3

class Config:
    def __init__(self):
        self.lr = LEARNING_RATE
        self.rs = REGULARIZATION_STRENGTH
        self.num_epochs = TRAINING_EPOCHS
        self.display_step = DISPLAY_STEP

        self.input_size = INPUT_SIZE
        self.hidden_size_1 = HIDDEN_SIZE_1
        self.hidden_size_2 = HIDDEN_SIZE_2
        self.output_size = OUTPUT_SIZE
