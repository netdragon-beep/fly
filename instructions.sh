conda activate fly

nvidia -smi

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

  $env:NO_PROXY = "127.0.0.1,localhost"  



   https://www.mdpi.com/2504-446X/8/10/562

   https://blog.csdn.net/longtengzhangjie/article/details/133266243 

  # 核心组件说明：

  # 1. 威胁评估 (threat_assessment.py)
  #   - 7维特征：距离、高度、速度、航向、俯仰、滚转、战斗状态
  #   - 归一化到[-1, 1]范围
  # 2. 注意力网络 (attention_network.py)
  #   - 4头多头注意力机制
  #   - TAPPOActor: 策略网络输出目标选择概率
  #   - TAPPOCritic: LSTM价值网络处理时序特征
  # 3. PPO训练器 (tappo_trainer.py)
  #   - 16步经验序列
  #   - GAE优势估计
  #   - PPO-Clip目标函数
  #   - 动态奖励：命中率 + 导弹效益比
  # 4. 行为树集成 (tappo_attack.py)
  #   - ActionTAPPOAttack: 攻击动作节点
  #   - 支持单机开火和双机协同开火
  #   - 与SmartFireControl火控系统结合

  # 使用方法：

  # # 训练
  # python train_tappo.py --mode train --episodes 1000 --device auto

  # # 评估
  # python train_tappo.py --mode eval --model checkpoints/best_model.pt --episodes 100

  # 在行为树中使用：
  # from tappo import ActionTAPPOAttack, create_tappo_agent

  # # 创建攻击动作节点
  # attack_action = ActionTAPPOAttack(
  #     model_path='tappo/checkpoints/best_model.pt',
  #     use_coordinated_fire=True
  # )

   
   python train.py --episodes 50000 --num-envs 64 --batch-size 4096 --device cuda