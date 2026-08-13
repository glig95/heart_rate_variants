"""
Start the program.

Open a terminal in this folder and run:

    python run_gui.py

If that says a package is missing, install the requirements first:

    pip install -r requirements.txt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    """Check that the needed packages are installed, then open the window."""
    missing = []
    for package in ["numpy", "pandas", "scipy", "matplotlib", "sklearn", "xgboost"]:
        try:
            __import__(package)
        except ImportError:
            missing.append("scikit-learn" if package == "sklearn" else package)
    if missing:
        print("These packages need to be installed first:", ", ".join(missing))
        print("Run:  pip install -r requirements.txt")
        return 1

    try:
        import tkinter  # noqa: checked here so the message is friendly
    except ImportError:
        print("Python was built without tkinter, which draws the window.")
        print("On macOS, installing Python from python.org includes it.")
        print("You can still run the analysis with examples/run_from_command_line.py")
        return 1

    from hrv.gui import main as open_window
    open_window()
    return 0


if __name__ == "__main__":
    sys.exit(main())
