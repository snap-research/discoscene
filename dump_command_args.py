# python3.7
"""Dumps available arguments of all commands (configurations).

This file parses the arguments of all commands provided in `configs/` and dump
the results as a json file. Each parsed argument includes the name, argument
type, default value, and the help message (description). The dumped file looks
like

{
    "command_1": {
        "type": "object",
        "properties": {
            "arg_group_1": {
                "type": "object",
                "properties": {
                    "arg_1": {
                        "is_recommended":  # true / false
                        "type":  # int / float / bool / str / json-string /
                                 # index-string
                        "default":
                        "description":
                    },
                    "arg_2": {
                        "is_recommended":
                        "type":
                        "default":
                        "description":
                    }
                }
            },
            "arg_group_2": {
                "type": "object",
                "properties": {
                    "arg_3": {
                        "is_recommended":
                        "type":
                        "default":
                        "description":
                    },
                    "arg_4": {
                        "is_recommended":
                        "type":
                        "default":
                        "description":
                    }
                }
            }
        }
    },
    "command_2": {
        "type": "object",
        "properties: {
            "arg_group_1": {
                "type": "object",
                "properties": {
                    "arg_1": {
                        "is_recommended":
                        "type":
                        "default":
                        "description":
                    }
                }
            }
        }
    }
}
"""

import sys
import json

from configs import CONFIG_POOL


def parse_args_from_config(config):
    """Parses available arguments from a configuration class.

    Args:
        config: The configuration class to parse arguments from, which is
            defined in `configs/`. This class is supposed to derive from
            `BaseConfig` defined in `configs/base_config.py`.
    """
    recommended_opts = config.get_recommended_options()
    args = dict()
    for opt_group, opts in config.get_options().items():
        args[opt_group] = dict(
            type='object',
            properties=dict()
        )
        for opt in opts:
            arg = config.inspect_option(opt)
            args[opt_group]['properties'][arg.name] = dict(
                is_recommended=arg.name in recommended_opts,
                type=arg.type,
                default=arg.default,
                description=arg.help
            )
    return args


def dump(configs, save_path):
    """Dumps available arguments from given configurations to target file.

    Args:
        configs: A list of configurations, each of which should be a
            class derived from `BaseConfig` defined in `configs/base_config.py`.
        save_path: The path to save the dumped results.
    """
    args = dict()
    for config in configs:
        args[config.name] = dict(type='object',
                                 properties=parse_args_from_config(config))
    with open(save_path, 'w') as f:
        json.dump(args, f, indent=4)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(f'Usage: python {sys.argv[0]} SAVE_PATH')
    dump(CONFIG_POOL, sys.argv[1])
