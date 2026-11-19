from setuptools import setup, find_packages
from pathlib import Path
this_directory = Path(__file__).parent
LONG_DESCRIPTION = (this_directory / "README.md").read_text()

VERSION = '0.0.7'
DESCRIPTION = 'GroupMultiNeSS package'

setup(
    name="GroupMultiNeSS",
    version=VERSION,
    author="Alexander Kagan",
    author_email="<amkagan@umich.edu>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    python_requires='<3.12',
    install_requires=["numpy<2",
                      "scipy",
                      "typing",
                      "joblib",
                      "matplotlib",
                      "seaborn",
                      "statsmodels",
                      "scikit-learn>=1.2"],

    keywords=['python', 'multiplex networks', 'multiness', 'latent space models'],

    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
