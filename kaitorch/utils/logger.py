from typing import Optional
from datetime import datetime
from pathlib import Path


class Logger:
    r'''

    #### Args:
    - name: a special name for logging file.
    - level: the lowest level to be accepted. It works when called as a method.

    #### Methods:
    - __call__: Do logging.
    - flush: Write all into the logging file.
    - debug: Log at debugging level.
    - info: Log at information level.
    - warn: Log at warning level.
    - error: Log at error level.

    ## __call__
    #### Args:
    - s: message.
    - level
    - verbose: Whether to print on the console.

    #### Returns:
    None

    '''

    DEBUG = 0
    INFO = 1
    WARN = 2
    ERROR = 3

    def __init__(self, name: Optional[str] = None, level: int = ERROR) -> None:
        root = Path.cwd() / 'log'
        root.mkdir(exist_ok=True)
        if name is None:
            path = root.joinpath(
                datetime.strftime(datetime.today(), '%Y%m%d%H%M%S') +
                '.log'
            )
        else:
            path = root.joinpath(
                name + '_' +
                datetime.strftime(datetime.today(), '%Y%m%d%H%M%S') +
                '.log'
            )
        self._f = open(path, 'a')
        self._format = '{time} - {level}: {msg}'
        self._level = level

    def __del__(self):
        self._f.close()

    def _write_f(self, s: str) -> None:
        self._f.write(s + '\n')
        self._f.flush()

    def __call__(
        self, s: str, level: int = INFO, verbose: bool = True
    ) -> None:
        if level < self._level:
            return
        match level:
            case self.INFO:
                self.info(s, verbose)
            case self.WARN:
                self.warn(s, verbose)
            case self.DEBUG:
                self.debug(s, verbose)
            case self.ERROR:
                self.error(s, verbose)
            case _:
                raise ValueError(f'an invalid level ({level}).')

    def debug(self, s: str, verbose: bool = True) -> None:
        r'''Log at debugging level.

        ### Args:
            - s: message.
            - verbose: Whether to print on the console.

        '''
        s = self._format.format(
            time=datetime.strftime(datetime.today(), '%Y/%m/%d %H:%M:%S'),
            level='DEBUG',
            msg=s
        )
        if verbose:
            print(s)
        self._write_f(s)

    def info(self, s: str, verbose: bool = True) -> None:
        r'''Log at information level.

        #### Args:
        - s: message.
        - verbose: Whether to print on the console.

        '''
        s = self._format.format(
            time=datetime.strftime(datetime.today(), '%Y/%m/%d %H:%M:%S'),
            level='INFO',
            msg=s
        )
        if verbose:
            print(s)
        self._write_f(s)

    def warn(self, s: str, verbose: bool = True) -> None:
        r'''Log at warning level.

        #### Args:
        - s: message.
        - verbose: Whether to print on the console.

        '''
        s = self._format.format(
            time=datetime.strftime(datetime.today(), '%Y/%m/%d %H:%M:%S'),
            level='WARNING',
            msg=s
        )
        if verbose:
            print('\033[33m' + s + '\033[0m')
        self._write_f(s)

    def error(self, s: str, verbose: bool = True) -> None:
        r'''Log at error level.

        #### Args:
        - s: message.
        - verbose: Whether to print on the console.

        '''
        s = self._format.format(
            time=datetime.strftime(datetime.today(), '%Y/%m/%d %H:%M:%S'),
            level='ERROR',
            msg=s
        )
        if verbose:
            print('\033[31m' + s + '\033[0m')
        self._write_f(s)
