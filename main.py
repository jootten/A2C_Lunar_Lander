from coordinator import Coordinator
from tensorboard import program
import ray
import argparse

def main(): 
    # argument parser to specify hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="training run", action="store_true")
    parser.add_argument("--test", help="test run", action="store_true")
    parser.add_argument("--network_type", default="mlp", help="""type of the policy network ["mlp", "gru"]""", type=str)
    parser.add_argument("--num_agents", default=8, help="number of environments and agents running in parallel", type=int)
    parser.add_argument("--num_steps", default=32, help="number of steps on each environment for every update", type=int)
    parser.add_argument("--environment", default="LunarLanderContinuous-v2", help="gym environment type", type=str)
    args = parser.parse_args()

    # Launch tensorboard
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', './logs'])
    tb.launch()

    # Initialize ray
    ray.init(
    memory= 1024 * 512 * 200,
    object_store_memory=1024 * 1024 * 1000
    )

    if args.train:
        # start training run with given hyperparameters
        coord = Coordinator(
            num_agents=args.num_agents, 
            network=args.network_type, 
            env_name=args.environment,
            num_steps=args.num_steps)
        coord.train()

    if args.test:
        # start run with latest model checkpoint
        from test import test_run
        test_run(
            network=args.network_type, 
            environment=args.environment
            )

    ray.shutdown()

if __name__ == '__main__':
    main()