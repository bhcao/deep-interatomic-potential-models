.. _training_io_handling:

IO handling during training
===========================

.. module:: dipm.training.training_io_handler

    .. autoclass:: TrainingIOHandler

        .. automethod:: __init__

        .. automethod:: attach_logger

        .. automethod:: log

        .. automethod:: save_checkpoint

        .. automethod:: save_dataset_info

        .. automethod:: restore_training_state

    .. autoclass:: LogCategory

.. module:: dipm.training.configs
    :noindex:

    .. autoclass:: TrainingIOHandlerConfig

        .. automethod:: __init__

.. module:: dipm.training.loggers.command_line

    .. autoclass:: LineLogger

        .. automethod:: __init__

        .. automethod:: __call__

    .. autoclass:: TableLogger

        .. automethod:: __init__

        .. automethod:: __call__

.. module:: dipm.training.loggers.visual_tools

    .. autoclass:: TensorBoardLogger

        .. automethod:: __init__

        .. automethod:: __call__

    .. autoclass:: WandbLogger

        .. automethod:: __init__

        .. automethod:: __call__
