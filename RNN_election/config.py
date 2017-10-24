INPUT_SIZE = 5
OUTPUT_SIZE = 5
STATE_SIZE = 30

LEARNING_RATE = 0.1
TRAINING_EPOCHS = 20000
DATA_SIZE = 1
DISPLAY_STEP = 1000

class Config():
    def __init__(self):
        self.fan_in = INPUT_SIZE
        self.fan_out = OUTPUT_SIZE
        self.state_size = STATE_SIZE

        self.lr = LEARNING_RATE
        self.num_epochs = TRAINING_EPOCHS
        self.display_step = DISPLAY_STEP
