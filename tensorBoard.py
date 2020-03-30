from tensorboard import program
from pathlib import Path

memory_dir = Path.cwd() / 'runs'
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', str(memory_dir), '--samples_per_plugin', 'scalars=10000'])
url = tb.launch()
print('TensorBoard at ' + url)
input("Press Enter to quit...")