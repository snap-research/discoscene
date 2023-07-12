# python3.7
"""Script to convert officially released models to match this repository."""

import os
import argparse

from converters import build_converter


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(description='Convert pre-trained models.')
    parser.add_argument('model_type', type=str,
                        choices=['pggan', 'stylegan', 'stylegan2',
                                 'stylegan2ada_tf', 'stylegan2ada_pth',
                                 'stylegan3'],
                        help='Type of the model to convert.')
    parser.add_argument('--source_model_path', type=str, required=True,
                        help='Path to load the model for conversion.')
    parser.add_argument('--target_model_path', type=str, required=True,
                        help='Path to save the converted model.')
    parser.add_argument('--forward_test_num', type=int, default=10,
                        help='Number of samples used for forward test. '
                             '(default: %(default)s)')
    parser.add_argument('--backward_test_num', type=int, default=0,
                        help='Number of samples used for backward test. '
                             '(default: %(default)s)')
    parser.add_argument('--save_test_image', action='store_true',
                        help='Whether to save the intermediate image in '
                             'forward test. (default: False)')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Learning rate used in backward test. '
                             '(default: %(default)s)')
    parser.add_argument('--verbose_log', action='store_true',
                        help='Whether to print verbose log. (default: False)')
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    if os.path.exists(args.target_model_path):
        raise SystemExit(f'File `{args.target_model_path}` has already '
                         f'existed!\n'
                         f'Please specify another path.')

    converter = build_converter(args.model_type, verbose_log=args.verbose_log)
    converter.run(src_path=args.source_model_path,
                  dst_path=args.target_model_path,
                  forward_test_num=args.forward_test_num,
                  backward_test_num=args.backward_test_num,
                  save_test_image=args.save_test_image,
                  learning_rate=args.learning_rate)


if __name__ == '__main__':
    main()
