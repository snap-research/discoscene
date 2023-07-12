# python3.7
"""Unit test for logger."""

import os
import time

from . import build_logger

__all__ = ['test_logger']

_TEST_DIR = 'logger_test'


def test_logger(test_dir=_TEST_DIR):
    """Tests loggers."""
    print('========== Start Logger Test ==========')

    os.makedirs(test_dir, exist_ok=True)

    for logger_type in ['normal', 'rich', 'dummy']:
        for indent_space in [2, 4]:
            for verbose_log in [False, True]:
                if logger_type == 'normal':
                    class_name = 'Logger'
                elif logger_type == 'rich':
                    class_name = 'RichLogger'
                elif logger_type == 'dummy':
                    class_name = 'DummyLogger'

                print(f'===== '
                      f'Testing  `utils.logger.{class_name}` '
                      f' (indent: {indent_space}, verbose: {verbose_log}) '
                      f'=====')
                logger_name = (f'{logger_type}_logger_'
                               f'indent_{indent_space}_'
                               f'verbose_{verbose_log}')
                logger = build_logger(
                    logger_type,
                    logger_name=logger_name,
                    logfile=os.path.join(test_dir, f'test_{logger_name}.log'),
                    verbose_log=verbose_log,
                    indent_space=indent_space)
                logger.print('print log')
                logger.print('print log,', 'log 2')
                logger.print('print log (indent level 0)', indent_level=0)
                logger.print('print log (indent level 1)', indent_level=1)
                logger.print('print log (indent level 2)', indent_level=2)
                logger.print('print log (verbose `False`)', is_verbose=False)
                logger.print('print log (verbose `True`)', is_verbose=True)
                logger.debug('debug log')
                logger.info('info log')
                logger.warning('warning log')
                logger.init_pbar()
                task_1 = logger.add_pbar_task('Task 1', 500)
                task_2 = logger.add_pbar_task('Task 2', 1000)
                for _ in range(1000):
                    logger.update_pbar(task_1, 1)
                    logger.update_pbar(task_2, 1)
                    time.sleep(0.002)
                logger.close_pbar()
                print('Success!')

    print('========== Finish Logger Test ==========')
