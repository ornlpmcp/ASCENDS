from setuptools import setup
 
setup(
    name='ascends-toolkit',
    version='0.4.0',
    description='ASCENDS: Advanced data SCiENce toolkit for Non-Data Scientists',
    long_description='',
    url='https://github.com/liza183/ascends-toolkit',
    author='Matt Sangkeun Lee, Dongwon Shin, Jian Peng',
    author_email='lees4@ornl.gov',
    license='MIT License',
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
    ],
    packages=[],
    py_modules=['ascends'],
    python_requires='>=3.2',
    install_requires=[
        'matplotlib',
        'numpy',
        'pandas',
        'tensorflow',
        'keras',
	'tornado',
        'scikit-learn',
        'minepy',
	'np_utils',
    ],  
    scripts=['train.py','predict.py'],
)
