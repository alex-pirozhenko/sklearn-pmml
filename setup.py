from setuptools import setup, Command


class PyTest(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import subprocess
        import sys
        errno = subprocess.call([sys.executable, 'runtests.py'])
        raise SystemExit(errno)

setup(
    name='sklearn-pmml',
    version='0.1.2',
    packages=['sklearn_pmml', 'sklearn_pmml.convert'],
    install_requires=[
        "pyxb",
        "scikit-learn",
        "pandas",
        "scipy",
        "pytest",
        "lxml",
        "enum34",
    ],
    cmdclass={'test': PyTest},
    url='https://github.com/alex-pirozhenko/sklearn-pmml',
    license='MIT',
    author='Alex Pirozhenko',
    author_email='apirozhenko@pulsepoint.com',
    description='A library that allows serialization of SciKit-Learn estimators into PMML'
)
