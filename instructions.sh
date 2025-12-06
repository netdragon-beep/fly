conda activate fly



  conda run -n fly python qyPython/env/agent/VEB-RL/train.py \
    --real \
    --population 100 \
    --elite-size 20 \
    --env-workers 4 \
    --episodes-per-elite 4 \
    --rl-batch-size 768 \
    --rl-updates 250 \
    --buffer-capacity 200000 \
    --target-update-freq 5 \
    --epsilon-decay 0.99 \
    --save-best-interval 10 \
    --load-pop ../checkpoints/veb/final.npz