#!/usr/bin/env python3.7
"""Simple CLI interface to StressDictionary.

Run `stresses_dictionary` without arguments to start interactive mode.

Usage:
    stresses_dictionary [<word> [<word> ...]]

Options:
    -h --help     Show this screen.
    --version     Show version.
"""
import docopt
import sys
import stress_dictionary as sd
from prompt_toolkit import prompt



stresser = None


def show_stresses(text):
    """ Sets stresses for given `text` and outputs it to stdout.
    Also creates StressDictionary if needed.

    1. text - text for set stresses
    """
    global stresser
    if not stresser:
        stresser = sd.StressDictionary()
    print(stresser.stress(text))


def interactive():
    """ Starts interactive mode
    """
    print("StressDictionary interactive mode.")
    print("Type `q` to quit.")
    try:
        while True:
            text = prompt("Your text: ")
            if text == 'q':
                break
            show_stresses(text)
    except EOFError:
        pass


def main():
    """ Parse arguments and fire selected mode
    """
    args = docopt.docopt(__doc__, version=sd.__version__)
    text = ' '.join(args['<word>'])
    if text:
        show_stresses(text)
    else:
        interactive()


if __name__ == '__main__':
    main()
