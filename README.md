# Global setup for I-EMIC using OMUSE #

### MacOS Installation Instructions ###

XCode commandline tool should be installed, this can be done via the following
terminal command:

    xcode-select --install


The simplest method for installing the prerequisites if via Homebrew
(https://brew.sh/):

    brew tap nlesc/nlesc
    brew install omuse-iemic

After this is done `python-omuse` can be used to run the omuse-iemic code via
`python-omuse iemic_global.py`. Alternatively, you can create a new Python
virtualenv with access to the omuse-iemic install via `omuse-env DIR`

### Ubuntu Linux Installation Instructions ###

The following command can be used to install all prerequisite packages:

    sudo apt-get install gfortran libopenblas-dev libhdf5-openmpi-dev libgsl0-dev \
          cmake libfftw3-3 libfftw3-dev libmpfr6 libmpfr-dev libnetcdf-dev \
          libnetcdff-dev libptscotch-dev trilinos-all-dev libslicot-dev

Create a new virtualenv:

    python3 -m venv DIR

Activate the virtualenv:

    . DIR/bin/activate

Install/update necessary packages:

    python -m pip install --upgrade pip setuptools wheel setuptools_scm
    pip install matplotlib omuse-iemic

The code can then be run using:

    python iemic_global.py
