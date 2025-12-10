"""
双机协同开火数据采集启动脚本

使用真实仿真环境采集训练数据

使用方法:
1. 确保仿真环境已启动
2. 运行: python run_data_collection.py
3. 数据会自动保存到 data/ 目录
4. 采集完成后运行 train_with_simulation.py 进行训练
"""

import os
import sys
import time
import argparse

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 切换工作目录到项目根目录
os.chdir(project_root)


def run_single_instance(args):
    """单实例运行数据采集"""
    print("=" * 60)
    print("双机协同开火 - 数据采集模式 (单实例)")
    print("=" * 60)

    try:
        import config
        from env.env import Env
        from env.agent.策略树.actions.粒子群双机开火决策.data_collection_agent import DualFireDataCollectionAgent

        # 创建采集智能体
        red_agent = DualFireDataCollectionAgent('red', '红方数据采集')

        # 蓝方使用普通AI或None
        blue_agent = None
        if args.blue_agent:
            from env.agent.策略树.agent import BTDemoAgent
            blue_agent = BTDemoAgent('blue', '蓝方AI')

        print(f"\n配置:")
        print(f"  红方: DualFireDataCollectionAgent")
        print(f"  蓝方: {'BTDemoAgent' if blue_agent else 'None'}")
        print(f"  自动保存间隔: {args.save_interval} 样本")

        # 运行仿真
        env = Env()
        env.run(red_agent, blue_agent)

        # 保存最终数据
        red_agent.save_data()

        print("\n数据采集完成!")
        print(f"总样本数: {len(red_agent.samples)}")

    except ImportError as e:
        print(f"\n错误: 无法导入必要模块: {e}")
        print("请确保从项目根目录运行，或者正确配置PYTHONPATH")
        return


def run_multi_instance(args):
    """多实例并行运行数据采集"""
    print("=" * 60)
    print("双机协同开火 - 数据采集模式 (多实例并行)")
    print("=" * 60)

    try:
        import config
        from env.multi_env import auto_engage_main_multi
        from env.agent.策略树.actions.粒子群双机开火决策.data_collection_agent import DualFireDataCollectionAgent

        # 修改config以支持多次运行
        original_run_times = config.run_times
        config.run_times = args.episodes

        print(f"\n配置:")
        print(f"  并行实例数: {config.generate_task_num}")
        print(f"  运行轮数: {args.episodes}")
        print(f"  自动保存间隔: {args.save_interval} 样本")

        # 创建采集智能体
        red_agent = DualFireDataCollectionAgent('red', '红方数据采集')
        red_agent.auto_save_interval = args.save_interval

        # 蓝方使用普通AI
        blue_agent = None
        if args.blue_agent:
            from env.agent.策略树.agent import BTDemoAgent
            blue_agent = BTDemoAgent('blue', '蓝方AI')

        # 运行多实例
        auto_engage_main_multi(red_agent, blue_agent)

        # 恢复config
        config.run_times = original_run_times

        # 保存最终数据
        red_agent.save_data()

        print("\n数据采集完成!")
        print(f"总样本数: {len(red_agent.samples)}")

    except ImportError as e:
        print(f"\n错误: 无法导入必要模块: {e}")
        print("请确保从项目根目录运行")
        return


def merge_data_files(args):
    """合并多个数据文件"""
    import json
    import glob

    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    pattern = os.path.join(data_dir, 'dual_fire_sim_*.json')
    files = glob.glob(pattern)

    if not files:
        print(f"没有找到数据文件: {pattern}")
        return

    print(f"找到 {len(files)} 个数据文件")

    all_samples = []
    for filepath in files:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        all_samples.extend(data.get('samples', []))
        print(f"  {os.path.basename(filepath)}: {len(data.get('samples', []))} 样本")

    # 保存合并后的数据
    merged_filepath = os.path.join(data_dir, 'dual_fire_merged.json')
    merged_data = {
        'source': 'simulation_merged',
        'num_samples': len(all_samples),
        'samples': all_samples
    }

    with open(merged_filepath, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

    print(f"\n合并完成: {merged_filepath}")
    print(f"总样本数: {len(all_samples)}")


def main():
    parser = argparse.ArgumentParser(description='双机协同开火数据采集')

    parser.add_argument('--mode', type=str, default='single',
                        choices=['single', 'multi', 'merge'],
                        help='运行模式: single(单实例), multi(多实例), merge(合并数据)')

    # 采集参数
    parser.add_argument('--episodes', type=int, default=10,
                        help='运行轮数 (多实例模式)')
    parser.add_argument('--save_interval', type=int, default=50,
                        help='自动保存间隔 (样本数)')

    # Agent参数
    parser.add_argument('--blue_agent', action='store_true',
                        help='是否使用蓝方AI (默认无)')

    args = parser.parse_args()

    if args.mode == 'single':
        run_single_instance(args)
    elif args.mode == 'multi':
        run_multi_instance(args)
    elif args.mode == 'merge':
        merge_data_files(args)


if __name__ == "__main__":
    main()
