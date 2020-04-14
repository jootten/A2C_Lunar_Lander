from coordinator import Coordinator
from tensorboard import program
import ray
import argparse

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=True, help="training (true) or test (false) run", type=bool)
    parser.add_argument("--network_type", default="mlp", help="""type of the policy network ["mlp", "lstm"]""", type=str)
    args = parser.parse_args()

    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', './logs'])
    tb.launch()

    ray.init(
    memory= 1024 * 512 * 200,
    object_store_memory=1024 * 1024 * 1000
    )

    coord = Coordinator(network=args.network_type)
    coord.run_for_episodes()

    ray.shutdown()
if __name__ == '__main__':
    main()