from setuptools import setup, find_packages, Extension
import pybind11

cpp_args = ['-std=c++11', '-stdlib=libc++', '-mmacosx-version-min=10.7']

fcc_module = Extension(
    'functionc',
    sources=['functionc.cpp'],
    include_dirs=[pybind11.get_include(), 
    
                  ".\\carma-0.5.2\\include",
                  
                  ".\\numpy\\core\\include",
                  
                  ".\\armadillo-10.6.2\\include"],
    language='c++',
    extra_compile_args=cpp_args,
    )

setup(
    name='topic_cluster_visualizer',
    version='0.2',
    description='',
    packages=find_packages(exclude=['tests']),
    install_requires = ['plotly >= 4.14.3', 'factor-analyzer >= 0.4.0'],
    ext_modules=[fcc_module],
)

