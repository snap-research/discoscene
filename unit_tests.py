# python3.7
"""Collects all unit tests."""

import argparse

from models.test import test_model
from utils.loggers.test import test_logger
from utils.visualizers.test import test_visualizer
from utils.parsing_utils import parse_bool


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(description='Run unit tests.')
    parser.add_argument('--result_dir', type=str,
                        default='work_dirs/unit_tests',
                        help='Path to save the test results. (default: '
                             '%(default)s)')
    parser.add_argument('--test_all', type=parse_bool, default=False,
                        help='Whether to run all unit tests. (default: '
                             '%(default)s)')
    parser.add_argument('--test_model', type=parse_bool, default=False,
                        help='Whether to run unit test on models. (default: '
                             '%(default)s)')
    parser.add_argument('--test_logger', type=parse_bool, default=False,
                        help='Whether to run unit test on loggers. (default: '
                             '%(default)s)')
    parser.add_argument('--test_visualizer', type=parse_bool, default=False,
                        help='Whether to do unit test on visualizers. '
                             '(default: %(default)s)')
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    if args.test_all or args.test_model:
        test_model()

    if args.test_all or args.test_logger:
        test_logger(args.result_dir)

    if args.test_all or args.test_visualizer:
        test_visualizer(args.result_dir)


if __name__ == '__main__':
    main()
