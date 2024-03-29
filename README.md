# Autonomy

`autonomy` is a Python package for computing measures of autonomy on small
artificial agents defined by their transition probability matrices.

**If you use this code, please cite the paper:**

---

Albantakis L (2021)
Quantifying the autonomy of structurally diverse
automata: a comparison of candidate measures. Forthcoming.

---

## Installation

Set up a Python 3 virtual environment and install with

```bash
pip install "git+https://github.com/Albantakis/autonomy.git"
```

Most functions in `autonomy` require [PyPhi](https://github.com/wmayner/pyphi),
the Python library for computing integrated information.

**Note:** this software is only supported on Linux and macOS. However, if you
use Windows, you can run it by using the [Anaconda
Python](https://www.anaconda.com/what-is-anaconda/) distribution and [installing
PyPhi with conda](https://anaconda.org/wmayner/pyphi):

```bash
conda install -c wmayner pyphi
```

I suggest following PyPhi's [detailed installation guide](https://github.com/wmayner/pyphi/blob/develop/INSTALLATION.rst).

## Documentation

### Getting started

The `Agent` object is the main object on which computations are performed. It
represents the causal model of the agent as a transition probability matrix.

Initializing an agent requires a transition probability matrix (TPM).  Providing
a connectivity matrix is optional.  Most functions also require an activity
attribute.  See example agents in `data` folder.  Finally, information about the
sensors, motors, and hidden units is required for most functions. See YAML files
in the `autonomy/phenotypes` folder.

Once an agent object is defined, a full structural analysis can be initialized
with the `fullStructuralAnalysis(agent)` function, and likewise for a full
dynamical, information-theoretical, or causal analysis. These functions output a
pandas dataframe with all computed values. **Note:** the full causal analysis
can be very time consuming.

An example notebook and agent can be found in the `autonomy/examples` folder.

## Contact

For any questions related to this code, and to report issues, please send an email to albantakis@wisc.edu.

<!-- ## Credit

### Please cite these papers if you use this code:

Mayner WGP, Marshall W, Albantakis L, Findlay G, Marchman R, Tononi G. (2018)
[PyPhi: A toolbox for integrated information
theory](https://doi.org/10.1371/journal.pcbi.1006343). PLOS Computational
Biology 14(7): e1006343. <https://doi.org/10.1371/journal.pcbi.1006343>

```
@article{mayner2018pyphi,
  title={PyPhi: A toolbox for integrated information theory},
  author={Mayner, William GP and Marshall, William and Albantakis, Larissa and Findlay, Graham and Marchman, Robert and Tononi, Giulio},
  journal={PLoS Computational Biology},
  volume={14},
  number={7},
  pages={e1006343},
  year={2018},
  publisher={Public Library of Science},
  doi={10.1371/journal.pcbi.1006343},
  url={https://doi.org/10.1371/journal.pcbi.1006343}
}
``` -->
