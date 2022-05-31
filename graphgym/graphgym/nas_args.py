import argparse
import sys


def parse_args():
    """Parses the arguments."""
    parser = argparse.ArgumentParser(
        description='Neural Architecture Search'
    )
    parser.add_argument(
        '--config_base',
        type=str,
        default=None
    )
    parser.add_argument(
        '--config_budget',
        type=str,
        default=None
    )
    parser.add_argument(
        '--config_grid',
        type=str,
        default=None
    )
    parser.add_argument(
        '--config_dir',
        type=str,
        default=None
    )
    parser.add_argument(
        '--metric',
        type=str,
        default=None
    )
    parser.add_argument(
        '--config_grid',
        type=str,
        default=None
    )
    parser.add_argument(
        '--config_grid',
        type=str,
        default=None
    )
    parser.add_argument(
        '--config_grid',
        type=str,
        default=None
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()
