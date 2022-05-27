from setuptools import setup, find_packages

setup(
    name='dollarstreet',
    version='0.0.1',
    author='Will Gaviria Rojas',
    author_email='will@coactive.ai',
    description='Dollar Street code',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'pandas',
        'numpy',
        'scikit-learn',
        'tqdm',
    ],
)
