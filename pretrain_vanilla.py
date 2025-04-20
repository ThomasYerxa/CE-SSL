from CE_SSL.trainers.train import train
from CE_SSL.utils.distributed import init_dist_node

from argparse import ArgumentParser
import submitit

parser = ArgumentParser()
parser.add_argument("--batch_size", type=int, default=2048)
parser.add_argument("--dataset", type=str, default="imagenet")
parser.add_argument("--n_aug", type=int, default=2)
parser.add_argument("--lr", type=float, default=0.6)
parser.add_argument("--tau", type=float, default=0.99)
parser.add_argument("--lmbda", type=float, default=0.0)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--num_workers", type=int, default=14)
parser.add_argument("--save_freq", type=int, default=20)
parser.add_argument(
    "--save_folder", type=str, default="./models/vanilla/"
)

parser.add_argument("--objective", type=str, default="Barlow")
parser.add_argument("--resnet_size", type=int, default=50)


parser.add_argument("--n_nodes", type=int, default=2)
parser.add_argument("--n_gpus", type=int, default=4)

args = parser.parse_args()


class SLURM_Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        # set up distributed environment
        init_dist_node(self.args)
        train(None, self.args)


# submitit job management
executor = submitit.AutoExecutor(folder="./slurm/vanilla/%j", slurm_max_num_timeout=30)

executor.update_parameters(
    mem_gb=128 * args.n_gpus,
    gpus_per_node=args.n_gpus,
    tasks_per_node=args.n_gpus,
    cpus_per_task=args.num_workers,
    nodes=args.n_nodes,
    name="MMCR",
    timeout_min=60 * 24 * 5,
    slurm_partition="gpu",
    constraint="a100-80gb",
    slurm_array_parallelism=512,
)

trainer = SLURM_Trainer(args)
job = executor.submit(trainer)
