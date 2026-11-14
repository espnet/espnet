from egs3.TEMPLATE.beats.run import build_parser, main
from espnet3.systems.beats.system import BeatsSystem

if __name__ == "__main__":
    main(args=build_parser().parse_args(), system_cls=BeatsSystem)
