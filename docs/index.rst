Welcome to PRISM's documentation!
================================

PRISM (Parallel Recursive Isomorphism Search for Molecules) is a high-performance algorithm for finding maximum common substructures in molecular graphs.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api
   advanced
   contributing
   changelog

Installation
-----------

To install PRISM, simply run:

.. code-block:: bash

   pip install prism-molecular

Quick Start
----------

Here's a simple example of using PRISM:

.. code-block:: python

   from prism import MolecularGraphMatcher

   # Initialize the matcher
   matcher = MolecularGraphMatcher()

   # Find maximum common substructure between two molecules
   result = matcher.find_maximum_common_subgraph(
       "CC(=O)O",     # Acetic acid
       "CCC(=O)O"     # Propionic acid
   )

   # Access the results
   print(f"Match size: {result.size}")
   print(f"Node mapping: {result.mapping}")
   print(f"Match score: {result.score}")
   print(f"Time taken: {result.match_time} seconds")

Features
--------

- Fast and accurate maximum common substructure (MCS) detection
- Support for multiple input formats:
   - SMILES strings
   - RDKit molecules
   - NetworkX graphs
   - Native MolecularGraph format
- Advanced node signature generation for improved matching
- Parallel processing capabilities
- Comprehensive molecular feature support
- Flexible matching parameters

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 