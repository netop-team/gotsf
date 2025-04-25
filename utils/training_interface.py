import sys


def progress_bar(iterable, prefix='', length=40, fill='█'):
    total = len(iterable)
    def print_bar(iteration):
        percent = ("{0:.1f}").format(100 * (iteration / float(total)))
        filled_length = int(length * iteration // total)
        bar = fill * filled_length + '-' * (length - filled_length)
        sys.stdout.write(f'\r{prefix} |{bar}| {percent}% Complete')
        sys.stdout.flush()
    for i, item in enumerate(iterable):
        yield item
        print_bar(i + 1)
    sys.stdout.write('\n')
    sys.stdout.flush()

