from pathlib import Path

from setuptools import find_packages, setup

_ROOT = Path(__file__).resolve().parent
_req = (_ROOT / "requirements.txt").read_text(encoding="utf-8")
_install_requires = [
    line.strip()
    for line in _req.splitlines()
    if line.strip() and not line.startswith("#")
]

setup(
    name="ehr-chronic-disease-risk-prediction-system",
    version="0.1.0",
    description="EHR-based chronic disease risk prediction (MVP)",
    license="MIT",
    license_files=["LICENSE"],
    packages=find_packages(
        exclude=["notebooks", "data", "tests"],
    ),
    python_requires=">=3.10",
    install_requires=_install_requires,
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
