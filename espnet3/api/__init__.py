"""The contracts of ESPnet3: what every system promises, in one place.

A system under ``espnet3/systems/<name>/`` may organise its training however
it likes; what it exposes to the rest of the toolkit is fixed here, so a
front end written against a contract works for every system that honours
it.

- :mod:`espnet3.api.inference` - calling a trained model: the task it
  performs, the fields it takes and returns, and how audio reaches it.

The candidates for what comes next, in the order they would pay off, are
the item a recipe's ``dataset/`` must return, the ``metric(...)`` call the
``measure`` stage makes, and the trainable model that fine-tuning loads
weights into. Each is a de facto convention today; it belongs here once it
is written down.
"""
