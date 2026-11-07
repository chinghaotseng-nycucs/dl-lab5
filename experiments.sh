#!/bin/bash
# The training commands of Lab5. Usage: bash experiments.sh <run>
#   Runs: speed, task1, task2, task3, no_double, no_per, no_multistep, task3_imp_a to c, task3_r1, task3_e1 to e7
# The settings here are my choices. The spec fixes only the tasks and the snapshot steps (spec 2 and 4.2.2).
set -e

# The same base for every Pong run, so that a difference between runs comes from the technique alone.
PONG="--env ALE/Pong-v5 --lr 0.0001 --batch-size 32 --memory-size 100000 --replay-start-size 20000"
PONG="$PONG --epsilon-decay 0.99998 --epsilon-min 0.02 --target-update-frequency 1000"
FULL="--double --per --n-step 3"
# task3_imp_b, the best run so far. Each experiment below changes one thing from it. Settings given later win.
BEST="$PONG $FULL --lr 0.00005 --max-env-steps 2500000 --snapshot-steps 600000,1000000,1500000,2000000,2500000"
# The final settings for the ablation and vanilla runs: task3_e1's, without the three techniques. 1M steps (spec 4.1).
FINAL="$PONG --lr 0.00005 --memory-size 500000 --max-env-steps 1000000"
# task3_e1e3 won (last-10 mean 19.01), so these are the settings for the ablation runs that the report uses.
FINAL2="$FINAL --adam-eps 0.00015"

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
  task3_r1)     # A second seed of task3_imp_b, to measure the noise between runs
    python run_experiment.py --name task3_r1 $BEST --seed 1 ;;
  task3_e1)     # Report E1: replay memory 500k
    python run_experiment.py --name task3_e1 $BEST --memory-size 500000 ;;
  task3_e2)     # Report E2: 1 update per 4 steps. lr x updates per step, epsilon and beta keep their env-step schedules
    python run_experiment.py --name task3_e2 $BEST --train-every 4 --lr 0.0002 --epsilon-decay 0.99992 \
      --per-beta-steps 250000 ;;
  task3_e3)     # Report E3: Adam epsilon 1.5e-4
    python run_experiment.py --name task3_e3 $BEST --adam-eps 0.00015 ;;
  task3_e4)     # Report E4: copy to the target network every 4000 updates
    python run_experiment.py --name task3_e4 $BEST --target-update-frequency 4000 ;;
  task3_e5)     # Report E5: limit the error used for priorities to 1
    python run_experiment.py --name task3_e5 $BEST --priority-clip 1 ;;
  task3_e6)     # Report E6: batch 64, 1 update per 2 steps. Replay count and lr x updates per step stay the same
    python run_experiment.py --name task3_e6 $BEST --batch-size 64 --train-every 2 --lr 0.0001 --epsilon-decay 0.99996 \
      --per-beta-steps 500000 ;;
  task3_e7)     # Report E7: a point ends the target, in training only
    python run_experiment.py --name task3_e7 $BEST --point-ends-target ;;
  task3_e1_s1)  # Report E1, second seed: is the larger memory's gain real?
    python run_experiment.py --name task3_e1_s1 $BEST --memory-size 500000 --seed 1 ;;
  task3_e1_s2)  # Report E1, third seed
    python run_experiment.py --name task3_e1_s2 $BEST --memory-size 500000 --seed 2 ;;
  task3_e1_s3)  # Report E1, fourth seed, because a seed that is lucky by 1M is worth 17% (spec 4.2.2)
    python run_experiment.py --name task3_e1_s3 $BEST --memory-size 500000 --seed 3 ;;
  task3_e1e2)   # Extra: E1 and E2 together. Fedus et al. found that more capacity and less replay help together
    python run_experiment.py --name task3_e1e2 $BEST --memory-size 500000 --train-every 4 --lr 0.0002 \
      --epsilon-decay 0.99992 --per-beta-steps 250000 ;;
  task3_e1e3)   # Extra: E1 and E3 together, because E3 had the next best score by 1M (17.7)
    python run_experiment.py --name task3_e1e3 $BEST --memory-size 500000 --adam-eps 0.00015 ;;
  task3_e1e3_s1) # E1 and E3 together, second seed: is its steadiness real?
    python run_experiment.py --name task3_e1e3_s1 $BEST --memory-size 500000 --adam-eps 0.00015 --seed 1 ;;
  task3_e1e3_s2) # E1 and E3 together, third seed
    python run_experiment.py --name task3_e1e3_s2 $BEST --memory-size 500000 --adam-eps 0.00015 --seed 2 ;;
  f1_no_double)    # Spec 4.1, ablation at the final settings (E1): without Double DQN
    python run_experiment.py --name f1_no_double $FINAL --per --n-step 3 ;;
  f1_no_per)       # Spec 4.1, ablation at the final settings (E1): without PER
    python run_experiment.py --name f1_no_per $FINAL --double --n-step 3 ;;
  f1_no_multistep) # Spec 4.1, ablation at the final settings (E1): with 1-step returns
    python run_experiment.py --name f1_no_multistep $FINAL --double --per ;;
  f2_vanilla)      # Spec 2.3 and Task 2: vanilla DQN at the final settings (E1), for a fair comparison
    python run_experiment.py --name f2_vanilla $FINAL ;;
  f1b_no_double)    # Spec 4.1, ablation at the final settings (E1 and E3): without Double DQN
    python run_experiment.py --name f1b_no_double $FINAL2 --per --n-step 3 ;;
  f1b_no_per)       # Spec 4.1, ablation at the final settings (E1 and E3): without PER
    python run_experiment.py --name f1b_no_per $FINAL2 --double --n-step 3 ;;
  f1b_no_multistep) # Spec 4.1, ablation at the final settings (E1 and E3): with 1-step returns
    python run_experiment.py --name f1b_no_multistep $FINAL2 --double --per ;;
  f2b_vanilla)      # Spec 2.3 and Task 2: vanilla DQN at the final settings (E1 and E3)
    python run_experiment.py --name f2b_vanilla $FINAL2 ;;
  *)
    echo "Usage: bash experiments.sh <run>. The runs are the names in this file."
    exit 1 ;;
esac
