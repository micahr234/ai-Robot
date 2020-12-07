import os
from pathlib import Path

memory_dir = Path.cwd() / 'runs'
res = os.system('tensorboard --logdir=' + str(memory_dir) + ' --reload_multifile true' + ' --reload_multifile true' + ' --max_reload_threads 3')