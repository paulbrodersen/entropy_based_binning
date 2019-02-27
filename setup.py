from setuptools import setup, find_packages
setup(
    name = 'entropy_based_binning',
    version = '0.0.1',
    description = 'Entropy based binning of discrete variables.',
    author = 'Paul Brodersen',
    author_email = 'paulbrodersen+ebb@gmail.com',
    url = 'https://github.com/paulbrodersen/entropy_based_binning',
    download_url = 'https://github.com/paulbrodersen/entropy_based_binning/archive/0.0.1.tar.gz',
    keywords = ['entropy', 'binning', 'partition problem'],
    classifiers = [ # Get strings from http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Mathematics'
    ],
    platforms=['Platform Independent'],
    packages=find_packages(),
    install_requires=['numpy'],
)
