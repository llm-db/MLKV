#include <Python.h> 

extern "C" {
    /* Creates a dummy empty module that can be imported from Python.
       The import from Python will load the .so consisting of this file
       in this extension, so that the TORCH_LIBRARY registrations
       in the kernels file are properly loaded. */
    PyObject* PyInit_sok(void)
    {
        static struct PyModuleDef module_def = {
            PyModuleDef_HEAD_INIT,
            "sok",   /* name of module */
            "SOK - Sparse Operation Kit",   /* module documentation */
            -1,     /* size of per-interpreter state of the module,
                       or -1 if the module keeps state in global variables. */
            NULL,   /* methods */
        };
        return PyModule_Create(&module_def);
    }
  }