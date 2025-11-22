.. _optimizer:

Optimizer
=========

.. module:: dipm.training.optimizer

    .. autoclass:: EMATracker

        .. automethod:: __init__

        .. automethod:: update

        .. automethod:: get_model

    .. autofunction:: get_default_mlip_optimizer

    .. autofunction:: get_mlip_optimizer_chain_with_flexible_base_optimizer

.. module:: dipm.training.configs
    :noindex:

    .. autoclass:: OptimizerConfig

        .. automethod:: __init__
