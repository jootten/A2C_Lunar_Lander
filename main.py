from coordinator import Coordinator
from tensorboard import program
import ray
import argparse

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="training run", action="store_true")
    parser.add_argument("--test", help="test run", action="store_true")
    parser.add_argument("--network_type", default="mlp", help="""type of the policy network ["mlp", "lstm"]""", type=str)
    parser.add_argument("--num_agents", default=8, help="number of environments and agents running in parallel", type=int)
    parser.add_argument("--environment", default="LunarLanderContinuous-v2", help="gym environment type", type=str)
    args = parser.parse_args()

    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', './logs'])
    tb.launch()

    ray.init(
    memory= 1024 * 512 * 200,
    object_store_memory=1024 * 1024 * 1000
    )

    if args.train:
        coord = Coordinator(
            num_agents=args.num_agents, 
            network=args.network_type, 
            env_name=args.environment)
        coord.run_for_episodes()

    if args.test:
        from test import test_run
        test_run(
            network=args.network_type, 
            environment=args.environment
            )

    ray.shutdown()

if __name__ == '__main__':
    main()