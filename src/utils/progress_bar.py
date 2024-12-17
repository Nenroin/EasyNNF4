import sys
import time
from typing import TextIO

import numpy as np


class ProgressBar:
    def __init__(
            self,
            total: int = None,
            pb_length: int = 20,
            # one or two chars
            # if fill='-' : |-------   |
            # if fill='->': |------>   |
            fill: chr or str = '->',
            unfilled: chr = ' ',
            out_stream: TextIO = None,
            prefix: str = '',
            postfix: str = '',
            unit: str = 'it',
            initial: int = 0,
    ):
        if out_stream is None:
            out_stream = sys.stderr

        self.pb_length = pb_length

        self.fill_start = str(fill)[0]
        self.fill_end = str(fill)[-1]
        self.unfilled = str(unfilled)[0]

        self.prefix = prefix
        self.postfix = postfix
        self.unit = unit
        self.out_stream = out_stream

        self.total = total
        self.initial = initial
        self.iteration = initial

        self.start_time = time.time()
        self.last_iteration_change_time = None
        self.last_print_time = None

    def update(self, increase=1):
        if not isinstance(increase, int) or increase <= 0:
            raise ValueError('Increase must be an integer greater than 0.')
        self.iteration += increase
        self.last_iteration_change_time = time.time()
        self.refresh()

    def refresh(self):
        bar = self.__get_progress_bar()
        time_section = self.__get_time_section()
        iterations_section = self.__get_iterations_section()
        completion_percentage = self.__get_completion_percentage()
        #self.display(f'{self.prefix}{completion_percentage} |{bar}| {iterations_section} {time_section} {self.postfix}')
        self.display(f'{self.prefix}{completion_percentage} {time_section} {self.postfix}')

    def display(self, msg=None):
        self.last_print_time = self.start_time
        self.out_stream.write(f'\r{msg}')
        self.out_stream.flush()

    def set_prefix(self, prefix, refresh=False):
        self.prefix = prefix
        if refresh:
            self.refresh()

    def set_postfix(self, postfix, refresh=False):
        self.postfix = postfix
        if refresh:
            self.refresh()

    def set_postfix_dict(self, data, refresh=False):
        self.postfix = ', '.join(f"{key}: {value}" for key, value in data.items())
        if refresh:
            self.refresh()

    def close(self, leave=True):
        print_chr = '\n' if leave else '\r'
        self.out_stream.write(f'{print_chr}')
        self.out_stream.flush()

    def __get_progress_bar(self):
        iteration = min(self.iteration, self.total)
        filled_length = int(self.pb_length * iteration // self.total)
        bar = ''
        if filled_length != 0:
            bar = self.fill_start * (filled_length - 1) + self.fill_end
        bar += self.unfilled * (self.pb_length - filled_length)
        return bar

    def __get_time_section(self):
        displayed_time_to_finish = self.__format_time(self.__get_time_to_finish())
        displayed_passed_time = self.__format_time(self.__get_passed_time())

        average_iteration_duration = self.__get_average_iteration_duration()
        if average_iteration_duration == 0:
            iterations_per_second = np.nan
        else:
            iterations_per_second = 1 / self.__get_average_iteration_duration()
        return f'[{displayed_passed_time} <- {displayed_time_to_finish}, {iterations_per_second:.3f}{self.unit}/s]'

    def __get_iterations_section(self):
        total_digits_number = len(str(self.total))
        return f'{self.iteration:{total_digits_number}}{self.unit}/{self.total}{self.unit}'

    def __get_completion_percentage(self):
        percentage = int(self.iteration / self.total * 100)
        return f'{percentage:3}%'

    @classmethod
    def __format_time(cls, seconds):
        if seconds >= 3600:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            formatted_time = f"{hours:02}h{minutes:02}m"
        elif seconds >= 60:
            minutes = int(seconds // 60)
            seconds = int(seconds % 60)
            formatted_time = f"{minutes:02}m{seconds:02}s"
        else:
            formatted_time = f"{seconds:05.2f}s"

        return formatted_time

    def __get_time_to_finish(self):
        remaining_iterations = self.total - self.iteration
        remaining_time = self.__get_average_iteration_duration() * remaining_iterations
        return remaining_time

    def __get_passed_time(self):
        return self.last_iteration_change_time - self.start_time

    def __get_average_iteration_duration(self):
        return self.__get_passed_time() / (self.iteration - self.initial)
