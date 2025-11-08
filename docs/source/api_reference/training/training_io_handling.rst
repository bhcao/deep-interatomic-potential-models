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

.. module:: dipm.training.training_loggers

    .. autofunction:: log_metrics_to_table

    .. autofunction:: log_metrics_to_line

    .. autofunction:: convert_mse_to_rmse_in_logs
