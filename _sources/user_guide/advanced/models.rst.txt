.. _models:

Models
======

.. _model_init:

Create a model and predictor
--------------------------------

This section discusses how to initialize an force model for subsequent training.
If you are just interested in loading a pre-trained model for application in simulations,
please see the dedicated section :ref:`below <load_model>`.

Our force models exist in two abstraction levels:

* On the one hand, we have the pure neural networks,
  which are classes derived from
  :py:class:`ForceModel <dipm.models.force_model.ForceModel>`. As a general rule,
  these raw models take in as input a graph's edge vectors and node representations and
  output a vector of node energies.

* On the other hand, we wrap these models into force field predictors which take care
  of computing properties such as total energy, forces, or stress from the force model's
  output and themselves take a `jraph.GraphsTuple` object from the
  `jraph <https://jraph.readthedocs.io/en/latest/>`_ library as input. The flax module
  that implements this is
  :py:class:`ForceFieldPredictor <dipm.models.predictor.ForceFieldPredictor>`.

The library currently interfaces six force model architectures, i.e., force models
implementations:

* `MACE <https://arxiv.org/abs/2206.07697>`_
  (class: :py:class:`Mace <dipm.models.mace.models.Mace>`),
* `NequIP <https://www.nature.com/articles/s41467-022-29939-5>`_
  (class: :py:class:`Nequip <dipm.models.nequip.models.Nequip>`),
* `ViSNet <https://www.nature.com/articles/s41467-023-43720-2>`_
  (class: :py:class:`Visnet <dipm.models.visnet.models.Visnet>`),
* `LiTEN <https://arxiv.org/abs/2507.00884>`_
  (class: :py:class:`LiTEN <dipm.models.liten.models.LiTEN>`),
* `EquiformerV2 <https://openreview.net/forum?id=mCOBKZmrzD>`_
  (class: :py:class:`EquiformerV2 <dipm.models.equiformer_v2.models.EquiformerV2>`),
* `UMA <https://arxiv.org/abs/2506.23971>`_
  (class: :py:class:`UMA <dipm.models.uma.models.UMA>`).

These networks can be created from their configuration (listed in options of :ref:`training`)
and a :py:class:`DatasetInfo <dipm.data.dataset_info.DatasetInfo>` object
that one obtained after the :ref:`data processing step <get_dataset_info>`. For the
sake of simplified usage, the config objects can be directly accessed from the network
classes via their `.Config` attribute (see example below).

For example, to create a force field that uses MACE, one can simply execute:

.. code-block:: python

    from dipm.models import Mace, ForceFieldPredictor
    from flax import nnx

    dataset_info = _get_from_data_processing()  # placeholder

    # with default config
    mace = Mace(Mace.Config(), dataset_info, rngs=nnx.Rngs(0))
    force_field = ForceFieldPredictor(mace)

    # with modified config
    mace = Mace(Mace.Config(num_channels=64), dataset_info, rngs=nnx.Rngs(0))
    force_field = ForceFieldPredictor(mace)

Unlike `MLIP <https://github.com/instadeepai/mlip>`_, we use `flax.nnx` as our backend.
It's a pytorch-like api without the need to seperate parameters from the model.
We recommend to visit the
`flax nnx documentation <https://flax.readthedocs.io/en/stable/nnx_basics.html>`_ for more details.

Make predictions
----------------

We can run a prediction with an force field predictor like this:

.. code-block:: python

    graph = _get_jraph_graph_from_somewhere()  # placeholder
    force_field.eval()  # set to evaluation mode
    prediction = force_field(graph)

The ``prediction`` includes several properties and is a dataclass of type
:py:class:`Prediction <dipm.typing.prediction.Prediction>`. The properties other than
energy and forces are only predicted optionally
(see ``predict_stress`` argument of `ForceFieldPredictor`).

If the input ``graph`` object (type: ``jraph.GraphsTuple``) contains multiple subgraphs,
for example, if it represents a batch, we can get the energy and forces of the ``i``-th
subgraph like this:

.. code-block:: python

    # For i-th energy
    energy_i = float(prediction.energy[i])

    # For i-th forces
    num_nodes_before_i = sum(graph.n_node[j] for j in range(0, i))
    forces_i = prediction.forces[num_nodes_before_i : num_nodes_before_i + graph.n_node[i]]


**Important caveat:**

A :py:class:`ForceFieldPredictor <dipm.models.predictor.ForceFieldPredictor>` can only process
graphs (of type `jraph.GraphsTuple`) that have at least two subgraphs in them.
Calling the force field on a graph that is not formally a batch will result in a
`ValueError`. This means that if you are working with these graph objects directly,
make sure a single graph of interest is always batched with a minimal dummy graph.
We recommend to use the function
:py:func:`create_graph_from_chemical_system() <dipm.data.helpers.graph_creation.create_graph_from_chemical_system>`
to prepare graphs as this allows to pass the argument
`batch_it_with_minimal_dummy=True` for convenience. An example is shown below:

.. code-block:: python

    import numpy as np
    from dipm.data import ChemicalSystem
    from dipm.data.helpers import create_graph_from_chemical_system

    # Example H2O molecule:
    #   - H (Z=1) has specie index 0
    #   - O (Z=8) has specie index 3 (H, C, N come first)
    system = ChemicalSystem(
        atomic_numbers = np.array([1, 8, 1]),
        atomic_species = np.array([0, 3, 0]),
        positions = np.array([[-.5, .0, .0], [.0, .2, .0], [.5, .0, .0]]),
    )

    graph = create_graph_from_chemical_system(
        chemical_system=system,
        distance_cutoff_angstrom=5,
        batch_it_with_minimal_dummy=True,
    )

.. _load_model:

Load a model from a safetensors archive
---------------------------------------

To load a model (e.g., MACE) from our lightweight safetensors format that we ship our
pre-trained models with, you can use the function
:py:func:`load_model <dipm.utils.model_io.load_model>`:

.. code-block:: python

    from dipm.models import Mace
    from dipm.utils.model_io import load_model

    # The second argument is optional for built-in models.
    force_field = load_model("path/to/model.safetensors", Mace)

Subsequently, you can use the returned force field
(type: :py:class:`ForceFieldPredictor <dipm.models.predictor.ForceFieldPredictor>`) for
any downstream tasks.

.. _load_trained_model:

Load a trained model from an Orbax checkpoint
---------------------------------------------

To load a trained model from an `orbax <https://orbax.readthedocs.io/en/latest/>`_
checkpoint, one can use the
:py:func:`load_parameters_from_checkpoint() <dipm.models.params_loading.load_parameters_from_checkpoint>`
helper function:

.. code-block:: python

    from dipm.models import ForceField
    from dipm.models.params_loading import load_parameters_from_checkpoint

    initial_force_field = _create_initial_force_field()  # placeholder

    # Load parameters
    loaded_params = load_parameters_from_checkpoint(
        local_checkpoint_dir="path/to/checkpoint/directory",  # must be local
        initial_params=initial_force_field.params,
        epoch_to_load=157,
        load_ema_params=False,
    )

    # Create new force field with those loaded parameters
    force_field = ForceField(initial_force_field.predictor, loaded_params)
