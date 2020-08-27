from setuptools import setup, find_packages

packages = find_packages(where='src')

setup(
    name='source-ddc',
    version='0.0.1.dev1',
    author='DSOC',
    description='Algorithms for the estimation of Dynamic Discrete Choice models.',
    long_description='Algorithms for the estimation of Dynamic Discrete Choice models.',
    url='https://github.com/sansan-inc',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Education',
        'Topic :: Scientific/Engineering :: Mathematics'
    ],
    license='Apache Software License (http://www.apache.org/licenses/LICENSE-2.0)',
    keywords='economics econometrics structural estimation',
    python_requires='>=3',
    install_requires=[
        'numpy==1.18.4',
        'pandas==1.0.3',
        'scipy==1.4.1',
        'statsmodels==0.12.0rc0',
        'bidict==0.20.0'
    ],
    test_requires=[
        'pytest==6.0.1',
        'pytest-benchmark==3.2.3',
        'pytest-profiling==1.7.0'
    ]
)
