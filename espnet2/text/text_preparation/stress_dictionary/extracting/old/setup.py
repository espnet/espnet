import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()

__version__ = 0.1

setuptools.setup(
    name='wikipedia-accent-parser',
    version='0.1',
    author="Alexey Savenkov",
    author_email="alexey.savenkov7@gmail.com",
    description="Wikipedia Accent Parser package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords='nlp wikipedia accents',
    project_urls={
        'Source': '',
    },
    license='Private License TWIN 1.0',
    install_requires=[],
    include_package_data=True
)