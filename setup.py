import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hybrid-trace-clustering",
    version="0.0.1",
    author="Mehrad Abdollahi",
    author_email="mhr.abdollahi@gmail.com",
    description="Hybrid Trace clustering in Process Mining",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mehradabd/hybrid-trace-clustering",
    packages=setuptools.find_packages(),
    license='GPL 3.0',
    install_requires=[
        'pm4py',
        'pandas',
        'scipy',
        'numpy',
        'scikit-learn',
    ]
)
