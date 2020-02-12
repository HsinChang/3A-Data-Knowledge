from skmultiflow.trees import HoeffdingTree
from hoeffdingOptionTree import HOT
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from skmultiflow.data.file_stream import FileStream
import matplotlib as plt

plt.interactive(True)

dataset = "elec"

# 1. Create a stream

stream = FileStream(dataset+".csv", n_targets=1, target_idx=-1)
# 2. Prepare for use
stream.prepare_for_use()
# 2. Instantiate the HoeffdingTree classifier
h = [
    HoeffdingTree(),
    HOT()
]
# 3. Setup the evaluator

evaluator = EvaluatePrequential(pretrain_size=1000, max_samples=20000, show_plot=True,
                                metrics=['accuracy', 'kappa'], output_file='result_'+dataset+'.csv',
                                batch_size=1)
# 4. Run
evaluator.evaluate(stream=stream, model=h)