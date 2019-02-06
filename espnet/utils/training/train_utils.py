import chainer
import logging


def check_early_stop(trainer, epochs):
    """Checks if the training was stopped by an early stopping trigger and warns the user if it's the case

    :param trainer: The trainer used for training
    :param epochs: The maximum number of epochs
    """
    end_epoch = trainer.updater.get_iterator('main').epoch
    if end_epoch < (epochs - 1):
        logging.warning("Hit early stop at epoch " + str(
            end_epoch) + "\nYou can change the patience or set it to 0 to run all epochs")


def set_early_stop(trainer, args):
    """Sets the early stop trigger given the program arguments

    :param trainer: The trainer used for training
    :param args: The program arguments
    """
    patience = args.patience
    criterion = args.early_stop_criterion
    epochs = args.epochs
    mode = 'max' if 'acc' in criterion else 'min'
    if patience > 0:
        trainer.stop_trigger = chainer.training.triggers.EarlyStoppingTrigger(monitor=criterion,
                                                                              mode=mode,
                                                                              patients=patience,
                                                                              max_trigger=(epochs, 'epoch'))
