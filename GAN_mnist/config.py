class Config:
  def __init__(self):
    self.data_dir = '/data/mnist/'
    self.out_dir = 'out/'
    self.ckpt_dir = 'ckpt/'
    
    self.z_dim = 100
    self.h_dim = 128
    self.out_dim = 784

    self.num_epochs = 10000
    self.batch_size = 128
    self.display_step = 100
    self.save_step = 100
    self.lr = 1e-3

    self.load_checkpoint = True
