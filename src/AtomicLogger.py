from datetime import datetime
from enum import IntEnum
import json
from colorama import Fore

class LogLevel(IntEnum):
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3

    def get_color(self):
        if self == LogLevel.DEBUG:
            return Fore.BLUE
        elif self == LogLevel.INFO:
            return Fore.GREEN
        elif self == LogLevel.WARNING:
            return Fore.YELLOW
        elif self == LogLevel.ERROR:
            return Fore.RED
        else:
            raise ValueError(f'Unknown log level {self}')

    def __str__(self):
        if self == LogLevel.DEBUG:
            return 'DEBUG'
        elif self == LogLevel.INFO:
            return 'INFO'
        elif self == LogLevel.WARNING:
            return 'WARNING'
        elif self == LogLevel.ERROR:
            return 'ERROR'
        else:
            raise ValueError(f'Unknown log level {self}')

class AtomicLogger:
    def __init__(self, logfile=None, prints=False, level=LogLevel.DEBUG):
        # Note: if the logfile is None, then it will not be written to.
        # as long as prints is True, it will still print to stdout.
        self.unflushed_logs = []
        self.logfile = logfile
        self.prints = prints
        self.log_level = level

    def _format(self, t, fn_name, msg, log_level):
        msg = json.dumps(msg)   # this santizises things a bit
        msg =  f'[{str(log_level)}] {fn_name}@{t}: {msg}'
        return msg
    
    def add_log(self, fn_name, msg, log_level):
        time = datetime.now().isoformat(timespec='seconds')

        if self.prints:
            print(f'{log_level.get_color()}[{fn_name}]{Fore.RESET}@{time}: {msg}')
        if log_level >= self.log_level:
            self.unflushed_logs.append((time, fn_name, msg, log_level))
        if log_level == LogLevel.ERROR:
            self.flush()

    def add_log_flush(self, fn_name, msg, log_level):
        self.add_log(fn_name, msg, log_level)
        self.flush()
    
    def flush(self):
        if self.logfile is not None and self.unflushed_logs:
            s = '\n'.join(self._format(t, fn_name, msg, level) for t, fn_name, msg, level in self.unflushed_logs) + '\n'
            self.unflushed_logs = []  
            with open(self.logfile, 'a') as f:
                f.write(s)
