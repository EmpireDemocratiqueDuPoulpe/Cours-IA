import setuptools

setuptools.setup(
    name="mlsp",
    version="0.1",
    description="Common things used in 4MLSP",
    url="https://www.youtube.com/watch?v=yebo5ILBMC0",
    author="Empire Démocratique du Poulpe",
    author_email="alexis.lecomte@supinfo.com",
    install_requires=[
        "colorama",
        "pandas", "matplotlib",
        "numpy", "scipy", "scikit-learn", "mixed-naive-bayes", "nltk"
    ],
    packages=setuptools.find_packages(),
    zip_safe=False
)
