from egs3.TEMPLATE.openbeats.run import build_parser, main
from espnet3.systems.openbeats.system import OpenBeatsSystem

if __name__ == "__main__":
    main(args=build_parser().parse_args(), system_cls=OpenBeatsSystem)
