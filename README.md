# Lab5: Value-Based Reinforcement Learning

Deep Learning 535528, Fall 2026. Student ID 315551118.

## Files

From the TA:

- `dqn.py`: the template. It differs from the release only inside the TODOs and on the lines its NOTEs say to enable.
- `test_model.py`: the TA's Pong test script, unchanged.

Mine (extra, not required, unless noted):

- `run_experiment.py`: one training run. It adds what the template leaves out: the replay memory, the CartPole
  setup, 20-game evaluation, snapshots at exact steps, hashes and versions recorded during training, and resume.
- `experiments.sh`: every training command with its settings.
- `eval_snapshot.py`, `test_model_task1.py`, `test_model_task2.py`, `test_model_task3.py`: the evaluation command for
  each task (the spec requires a runnable command, section 4.2.2).
- `requirements.txt`: required by spec section 4.2.2.
- `make_plots.py`: comparison figures and step counts from the run logs.
- `smoke_test.py`: checks the code against the worked examples in my notes, then runs everything briefly.
- `pack_submission.py`: builds the ZIP that spec section 5 lists.

## How to run

```bash
pip install -r requirements.txt
python smoke_test.py                # a few minutes, no login needed

bash experiments.sh task1           # or task2, task3, no_double, no_per, no_multistep

python test_model_task1.py --model_path LAB5_315551118_task1.pt
python test_model_task2.py --model_path LAB5_315551118_task2.pt
python test_model_task3.py --model_path LAB5_315551118_task3_2500000.pt
```

Each evaluation plays 20 games on seeds 0 to 19 with no random actions, and prints one line per seed and the average.

To resume after a crash, start a new run from the last `resume.pt`, with the same settings:

```bash
python run_experiment.py --name task3_r1 --resume-from runs/task3/resume.pt <the settings of task3>
```
