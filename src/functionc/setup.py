from setuptools import setup, Extension
import pybind11

cpp_args = ['-std=c++11', '-stdlib=libc++', '-mmacosx-version-min=10.7']

fcc_module = Extension(
    'functionc',
    sources=['functionc.cpp'],
    include_dirs=[pybind11.get_include(), 
                  "C:\\Users\\Song1\\source\\repos\\Project3\\functionc\\carma-0.5.2\\include",
                  "C:\\Users\\Song1\\anaconda3\\Lib\\site-packages\\numpy\\core\\include",
                  "C:\\Users\\Song1\\source\\repos\\Project3\\functionc\\armadillo-10.6.2\\include"],
    language='c++',
    extra_compile_args=cpp_args,
    )

setup(
    name='functionc',
    version='1.0',
    description='Python package with functionc C++ extension (PyBind11)',
    ext_modules=[fcc_module],
)

