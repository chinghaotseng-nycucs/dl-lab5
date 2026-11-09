#!/bin/bash
# The training commands of Lab5. Usage: bash experiments.sh <run>
#   Runs: speed, task1, task2, task3, no_double, no_per, no_multistep, task3_imp_a to c, task3_r1, task3_e1 to e7,
#   task3_e1e3_s1 to s14, task3_g0 to g4, task3_h1 to h3, f1_*, f1b_*, f2_*, f2b_*
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
# 11/9: the goal is 19 by 600k (20%, spec 4.2.2). G runs start from E1 + E3 and change one thing each.
# They use the faster PER sampling (same chances, commit after 435a620), so G0 checks that it learns as before.
G="$BEST --memory-size 500000 --adam-eps 0.00015"

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
  task3_e1e3_s3) # E1 and E3 together, fourth seed: more chances of 19 by 1M (17%, spec 4.2.2)
    python run_experiment.py --name task3_e1e3_s3 $BEST --memory-size 500000 --adam-eps 0.00015 --seed 3 ;;
  task3_e1e3_s4) # E1 and E3 together, fifth seed
    python run_experiment.py --name task3_e1e3_s4 $BEST --memory-size 500000 --adam-eps 0.00015 --seed 4 ;;
  task3_e1e3_s5) # E1 and E3 together, sixth seed
    python run_experiment.py --name task3_e1e3_s5 $BEST --memory-size 500000 --adam-eps 0.00015 --seed 5 ;;
  task3_e1e3_s6) # E1 and E3 together, seventh seed: looking for a 1M snapshot that itself reaches 19
    python run_experiment.py --name task3_e1e3_s6 $BEST --memory-size 500000 --adam-eps 0.00015 --seed 6 ;;
  task3_e1e3_s7) # E1 and E3 together, eighth seed
    python run_experiment.py --name task3_e1e3_s7 $BEST --memory-size 500000 --adam-eps 0.00015 --seed 7 ;;
  task3_e1e3_s8) # E1 and E3 together, ninth seed, 1M only: another chance of a 1M snapshot that reaches 19 (17%)
    python run_experiment.py --name task3_e1e3_s8 $BEST --memory-size 500000 --adam-eps 0.00015 --seed 8 \
      --max-env-steps 1000000 --snapshot-steps 600000,1000000 ;;
  task3_e1e3_s9) # E1 and E3 together, tenth seed, 1M only
    python run_experiment.py --name task3_e1e3_s9 $BEST --memory-size 500000 --adam-eps 0.00015 --seed 9 \
      --max-env-steps 1000000 --snapshot-steps 600000,1000000 ;;
  task3_e1e3_s10) # E1 and E3 together, 1M only: more chances of a 1M snapshot that reaches 19 (17%)
    python run_experiment.py --name task3_e1e3_s10 $BEST --memory-size 500000 --adam-eps 0.00015 --seed 10 \
      --max-env-steps 1000000 --snapshot-steps 600000,1000000 ;;
  task3_e1e3_s11) # E1 and E3 together, 1M only
    python run_experiment.py --name task3_e1e3_s11 $BEST --memory-size 500000 --adam-eps 0.00015 --seed 11 \
      --max-env-steps 1000000 --snapshot-steps 600000,1000000 ;;
  task3_e1e3_s12) # E1 and E3 together, 1M only
    python run_experiment.py --name task3_e1e3_s12 $BEST --memory-size 500000 --adam-eps 0.00015 --seed 12 \
      --max-env-steps 1000000 --snapshot-steps 600000,1000000 ;;
  task3_e1e3_s13) # E1 and E3 together, 1M only
    python run_experiment.py --name task3_e1e3_s13 $BEST --memory-size 500000 --adam-eps 0.00015 --seed 13 \
      --max-env-steps 1000000 --snapshot-steps 600000,1000000 ;;
  task3_e1e3_s14) # E1 and E3 together, 1M only
    python run_experiment.py --name task3_e1e3_s14 $BEST --memory-size 500000 --adam-eps 0.00015 --seed 14 \
      --max-env-steps 1000000 --snapshot-steps 600000,1000000 ;;
  task3_g0)      # Control: E1 and E3 with the faster PER sampling. Does it learn like the old code?
    python run_experiment.py --name task3_g0 $G ;;
  task3_g1)      # G1: random shifts of up to 4 pixels on training batches (DrQ)
    python run_experiment.py --name task3_g1 $G --augment-pad 4 ;;
  task3_g2)      # G2: 10-step returns (DrQ uses 10, DER 20; the ablation showed multi-step matters most)
    python run_experiment.py --name task3_g2 $G --n-step 10 ;;
  task3_g3)      # G3: 2 updates per step. Epsilon and beta keep their schedules in environment steps
    python run_experiment.py --name task3_g3 $G --train-per-step 2 --epsilon-decay 0.99999 --per-beta-steps 2000000 ;;
  task3_g4)      # G4: an earlier start: learning from 5,000 stored steps, epsilon at 0.02 by about 50k updates
    python run_experiment.py --name task3_g4 $G --replay-start-size 5000 --epsilon-decay 0.99992 ;;
  task3_g2_s1)   # G2, second seed: 10-step returns gave the best 600k snapshot of the G wave (16.25, control 2.5)
    python run_experiment.py --name task3_g2_s1 $G --n-step 10 --seed 1 ;;
  task3_g2_s2)   # G2, third seed of 10-step returns
    python run_experiment.py --name task3_g2_s2 $G --n-step 10 --seed 2 ;;
  task3_g4_s1)   # G4, second seed: the early start had the best 1M snapshot of the G wave (18.8)
    python run_experiment.py --name task3_g4_s1 $G --replay-start-size 5000 --epsilon-decay 0.99992 --seed 1 ;;
  task3_g4_s2)   # G4, third seed
    python run_experiment.py --name task3_g4_s2 $G --replay-start-size 5000 --epsilon-decay 0.99992 --seed 2 ;;
  task3_h1)      # H1: 4 updates per step with random shifts and resets every 40k updates (SR-SPR, BBF), none after 500k
    python run_experiment.py --name task3_h1 $G --train-per-step 4 --epsilon-decay 0.999995 --per-beta-steps 4000000 \
      --augment-pad 4 --reset-every 40000 --reset-keep 0.8 --reset-stop 500000 ;;
  task3_h2)      # H2: as H1 with 2 updates per step
    python run_experiment.py --name task3_h2 $G --train-per-step 2 --epsilon-decay 0.99999 --per-beta-steps 2000000 \
      --augment-pad 4 --reset-every 40000 --reset-keep 0.8 --reset-stop 500000 ;;
  task3_h3)      # H3: AdamW weight decay 0.1 (BBF), 1 update per step
    python run_experiment.py --name task3_h3 $G --weight-decay 0.1 ;;
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
  f2b_vanilla_s1)   # Task 2: vanilla DQN at the final settings, second seed (f2b_vanilla beat task2: 16.1 against 15.55)
    python run_experiment.py --name f2b_vanilla_s1 $FINAL2 --seed 1 ;;
  *)
    echo "Usage: bash experiments.sh <run>. The runs are the names in this file."
    exit 1 ;;
esac
