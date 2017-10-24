TOTAL_STEP = 20000
DISPLAY_STEP = 100

INPUT_SIZE = 784
OUTPUT_SIZE = 10

BATCH_SIZE = 50

LEARNING_RATE = 1e-4

LOG_DIR = "./temp/logfile"


class Config():
    def __init__(self):
        self.total_step = TOTAL_STEP
        self.display_step = DISPLAY_STEP

        self.fan_in = INPUT_SIZE
        self.fan_out = OUTPUT_SIZE
        
        self.batch_size = BATCH_SIZE
        
        self.lr = LEARNING_RATE

        self.log_dir = LOG_DIR
