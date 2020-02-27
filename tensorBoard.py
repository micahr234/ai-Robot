from tensorboard import program
from pathlib import Path

memory_dir = Path.cwd() / 'runs'
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', str(memory_dir)])
url = tb.launch()

input("Press Enter to quit...")