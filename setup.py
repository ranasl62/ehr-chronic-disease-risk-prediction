from pathlib import Path

from setuptools import find_packages, setup

_ROOT = Path(__file__).resolve().parent
_req = (_ROOT / "requirements.txt").read_text(encoding="utf-8")
_install_requires = [
    line.strip()
    for line in _req.splitlines()
    if line.strip() and not line.startswith("#")
]
_readme = (_ROOT / "README.md").read_text(encoding="utf-8")

setup(
    name="ehr-chronic-disease-risk-prediction-system",
    version="1.0.0",
    description="Leakage-aware EHR clinical prediction research framework (OpenHealth working package)",
    long_description=_readme,
    long_description_content_type="text/markdown",
    author="Md Rana Hossain",
    author_email="support@larucare.com",
    url="https://github.com/ranasl62/ehr-chronic-disease-risk-prediction",
    project_urls={
        "Source": "https://github.com/ranasl62/ehr-chronic-disease-risk-prediction",
        "Issues": "https://github.com/ranasl62/ehr-chronic-disease-risk-prediction/issues",
        "Documentation": "https://github.com/ranasl62/ehr-chronic-disease-risk-prediction#readme",
        "Why": "https://github.com/ranasl62/ehr-chronic-disease-risk-prediction/blob/main/WHY_THIS_FRAMEWORK.md",
        "Limitations": "https://github.com/ranasl62/ehr-chronic-disease-risk-prediction/blob/main/LIMITATIONS.md",
        "Feedback": "https://github.com/ranasl62/ehr-chronic-disease-risk-prediction/blob/main/docs/HOW_IT_HELPS.md",
    },
    license="MIT",
    license_files=["LICENSE"],
    packages=find_packages(
        exclude=["notebooks", "data", "tests"],
    ),
    python_requires=">=3.10",
    install_requires=_install_requires,
    entry_points={
        "console_scripts": [
            "ehr-ai=openhealth.cli:main",
        ],
    },
    keywords=[
        "ehr",
        "electronic-health-records",
        "chronic-disease",
        "calibration",
        "shap",
        "temporal-leakage",
        "clinical-ml",
        "openhealth",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
)
