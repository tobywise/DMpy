Installing DMpy
"""""""""""""""

Windows
-------

First do

conda install mkl mkl-service libpython m2w64-toolchain

Then DMpy/theano afterwards

Certain things need to be in your path. This should happen when installing anaconda, however it is possible to set this up manually if not. In windows, you can add to the path by going to This PC > Properties > Advanced system settings > Environment variables

These paths should be added (where Anaconda2 is your python installation directory)

C:\Anaconda2
C:\Anaconda2\Library\mingw-w64\bin
C:\Anaconda2\Library\usr\bin
C:\Anaconda2\Library\bin
C:\Anaconda2\Scripts

Theano (which DMpy is built on top of) has a couple of requirements that need to be installed separately. This can be done easily at the terminal/command line:

conda install mkl-service libpython mingw


Linux
-----

conda install mkl theano pygpu



Errors during installation
--------------------------

"collect2.exe: error: ld returned 1 exit status"

This is an error originating from Theano and can be solved by installing the right packages.

conda install numpy scipy mkl mkl-service libpython

Or try updating everything

conda update --all


WARNING (theano.configdefaults): g++ not detected ! Theano will be unable to execute optimized C-implementations (for both CPU and GPU) and will default to Python implementations. Performance will be severely degraded. To remove this warning, set Theano flags cxx to an empty string.

This indicates that some dependencies are missing. Running the following at the terminal/command line should help:

conda install mkl mkl-service libpython mingw

WARNING (theano.tensor.blas): Using NumPy C-API based implementation for BLAS functions.

You probably installed DMpy/theano before dependencies. Uninstall everything then install in the correct order.

RuntimeError: To use MKL 2018 with Theano you MUST set "MKL_THREADING_LAYER=GNU" in your environement.

To fix this simply go to the command line (Windows) or terminal (OSX/Linux) and set the relevant environment variable.

set MKL_THREADING_LAYER=GNU

Or in windows create a new environment variable in the GUI


Or try installing MKL 2017 instead:
conda uninstall mkl=2018
conda install mkl=2017

Intel MKL FATAL ERROR: Cannot load mkl_intel_thread.dll

Not sure exactly what this means, but installing mkl 2017 seems to help.


Errors post-installation
------------------------

DMpy is built on top of Theano, and Theano is a rather temperamental package which (in my experience) occasionally decides to throw errors or refuse to import itself for no apparent reason. In the event of any Theano-related error, the simplest solution is often to reinstall Theano and some of its dependencies. As before, it is important to install Theano after the dependencies.

.. code-block:: bash

    conda uninstall mkl mkl-service m2w64-toolchain libpython theano

    conda install mkl mkl-service m2w64-toolchain libpython

    conda install theano


