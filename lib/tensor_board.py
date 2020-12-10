from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
import time


class tensor_board(torch.utils.tensorboard.SummaryWriter):

    def __init__(self, name):

        print('')
        tensor_board_dir = Path.cwd() / 'runs' / name / str(time.time())
        print('Creating Tensor Board')
        print('Log File: ' + str(tensor_board_dir))
        super().__init__(tensor_board_dir, max_queue=10000, flush_secs=60)