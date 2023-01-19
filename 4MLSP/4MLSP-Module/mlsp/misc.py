# #################################################################################################################### #
#       misc.py                                                                                                        #
#           The big box full of random things.                                                                         #
# #################################################################################################################### #

import shutil
from colorama import Fore


def print_title(title, char="#"):
    terminal_width = shutil.get_terminal_size().columns
    txt = f"{Fore.GREEN}{char * 4} {title} "

    print(f"\n{txt}{char * (terminal_width - len(txt))}")
