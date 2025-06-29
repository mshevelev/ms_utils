from setuptools import setup, find_packages

setup(
    name='ms_utils',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'bokeh',
        'holoviews',
        'panel',
        # Add any other dependencies here
    ],
)
