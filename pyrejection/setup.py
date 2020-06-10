from setuptools import setup, find_packages
setup(
    name='pyrejection',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        # List packages and versions you depend on.
        'scikit-learn==0.22.1',
        'arff==0.9',
        'plotly==4.5.0',
    ],
    extras_require={
        # Best practice to list non-essential dev dependencies here.
        'dev': [
            'mypy==0.770',
            'flake8==3.7.9',
            'pytest==5.2.2',
            'pytest-cov==2.8.1',
        ]
    }
)
