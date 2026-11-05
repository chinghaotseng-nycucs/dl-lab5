#!/bin/bash
# The training commands of Lab5. Usage: bash experiments.sh <run>
#   Runs: speed, task1, task2, task3, no_double, no_per, no_multistep
# The settings here are my choices. The spec fixes only the tasks and the snapshot steps (spec 2 and 4.2.2).
set -e

# The same base for every Pong run, so that a difference between runs comes from the technique alone.
PONG="--env ALE/Pong-v5 --lr 0.0001 --batch-size 32 --memory-size 100000 --replay-start-size 20000"
PONG="$PONG --epsilon-decay 0.99998 --epsilon-min 0.02 --target-update-frequency 1000"
FULL="--double --per --n-step 3"

case "$1" in
  speed)        # a short test: how many environment steps per second on this machine
    python run_experiment.py --name "speed_$(hostname)_$(date +%H%M)" $PONG $FULL --replay-start-size 2000 \
      --max-env-steps 20000 --wandb-mode disabled ;;
  task1)        # Spec 2.1. CartPole, target: an average above 480
    python run_experiment.py --name task1 --env CartPole-v1 --lr 0.001 --memory-size 50000 --replay-start-size 500 \
      --epsilon-decay 0.999 --epsilon-min 0.01 --target-update-frequency 200 --max-env-steps 50000 ;;
  task2)        # Spec 2.2. Vanilla DQN on Pong
    python run_experiment.py --name task2 $PONG --max-env-steps 1000000 ;;
  task3)        # Spec 2.3. All three techniques, up to 2.5M for the snapshots of spec 4.2.2
    python run_experiment.py --name task3 $PONG $FULL --max-env-steps 2500000 \
      --snapshot-steps 600000,1000000,1500000,2000000,2500000 ;;
  no_double)    # Spec 4.1, ablation: Task 3 without Double DQN
    python run_experiment.py --name no_double $PONG --per --n-step 3 --max-env-steps 1000000 ;;
  no_per)       # Spec 4.1, ablation: Task 3 without PER
    python run_experiment.py --name no_per $PONG --double --n-step 3 --max-env-steps 1000000 ;;
  no_multistep) # Spec 4.1, ablation: Task 3 with 1-step returns
    python run_experiment.py --name no_multistep $PONG --double --per --max-env-steps 1000000 ;;
  task3_imp_a)  # Improvement run 1 (stop rule: one change): Task 3 with 2 updates per environment step
    python run_experiment.py --name task3_imp_a $PONG $FULL --train-per-step 2 --max-env-steps 2500000 \
      --snapshot-steps 600000,1000000,1500000,2000000,2500000 ;;
  task3_imp_b)  # Improvement run 2 (one change): Task 3 with learning rate 0.00005
    python run_experiment.py --name task3_imp_b $PONG $FULL --lr 0.00005 --max-env-steps 2500000 \
      --snapshot-steps 600000,1000000,1500000,2000000,2500000 ;;
  task3_imp_c)  # Improvement run 3, the last (one change from imp_b): learning rate 0.00003
    python run_experiment.py --name task3_imp_c $PONG $FULL --lr 0.00003 --max-env-steps 2500000 \
      --snapshot-steps 600000,1000000,1500000,2000000,2500000 ;;
  *)
    echo "Usage: bash experiments.sh speed|task1|task2|task3|no_double|no_per|no_multistep|task3_imp_a|task3_imp_b|task3_imp_c"
    exit 1 ;;
esac
