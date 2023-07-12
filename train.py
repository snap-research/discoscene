# python3.7
"""Main function for model training."""
import warnings
warnings.filterwarnings("ignore")

import click

from configs import CONFIG_POOL
from configs import build_config
from runners import build_runner
from utils.dist_utils import init_dist
from utils.dist_utils import exit_dist


@click.group(name='Distributed Training',
             help='Train a deep model by choosing a command (configuration).',
             context_settings={'show_default': True, 'max_content_width': 180})
@click.option('--launcher', default='pytorch',
              type=click.Choice(['pytorch', 'slurm']),
              help='Distributed launcher.')
@click.option('--backend', default='nccl',
              type=click.Choice(['nccl', 'gloo', 'mpi']),
              help='Distributed backend.')
@click.option('--local_rank', type=int, default=0, hidden=True,
              help='Replica rank on the current node. This field is required '
                   'by `torch.distributed.launch`.')
def command_group(launcher, backend, local_rank):  # pylint: disable=unused-argument
    """Defines a command group for launching distributed jobs.

    This function is mainly for interaction with the command line. The real
    launching is executed by `main()` function, through `result_callback()`
    decorator. In other words, the arguments obtained from the command line will
    be passed to `main()` function. As for how the arguments are passed, it is
    the responsibility of each command of this command group. Please refer to
    `BaseConfig.get_command()` in `configs/base_config.py` for more details.
    """


@command_group.result_callback()
@click.pass_context
def main(ctx, kwargs, launcher, backend, local_rank):
    """Main function for distributed training.

    Basically, this function first initializes a distributed environment, then
    parses configuration from the command line, and finally sets up the runner
    with the parsed configuration for training.
    """
    _ = local_rank  # unused variable

    # Initialize distributed environment.
    init_dist(launcher=launcher, backend=backend)

    # Build configurations and runner.
    config = build_config(ctx.invoked_subcommand, kwargs).get_config()
    runner = build_runner(config)

    # Start training.
    runner.train()
    runner.close()

    # Exit distributed environment.
    exit_dist()


if __name__ == '__main__':
    # Append all available commands (from `configs/`) into the command group.
    for cfg in CONFIG_POOL:
        command_group.add_command(cfg.get_command())
    # Run by interacting with command line.
    command_group()  # pylint: disable=no-value-for-parameter
