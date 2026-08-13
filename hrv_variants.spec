# PyInstaller recipe for a standalone application.
#
# Build it on the machine it is meant to run on: PyInstaller cannot make a
# Windows program from a Mac or from Linux. On Windows, double-click
# build_windows.bat; on macOS or Linux, run
#
#     pip install pyinstaller
#     pyinstaller hrv_variants.spec
#
# The result is in dist/HeartRateVariants/. The data folder is copied in beside
# the program rather than sealed inside it, so the alignments, species file and
# trees can all be replaced without rebuilding.

import shutil
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

HERE = Path(SPECPATH)

# XGBoost ships a compiled library of its own (libxgboost.dll on Windows) that
# PyInstaller does not find by itself, and the program will not start without
# it. collect_all picks up that library along with everything else the package
# expects to find beside it.
xgb_datas, xgb_binaries, xgb_hidden = collect_all("xgboost")
sklearn_datas, sklearn_binaries, sklearn_hidden = collect_all("sklearn")

analysis = Analysis(
    [str(HERE / "run_gui.py")],
    pathex=[str(HERE)],
    binaries=xgb_binaries + sklearn_binaries,
    datas=xgb_datas + sklearn_datas,
    hiddenimports=[
        "sklearn.impute", "sklearn.preprocessing", "sklearn.feature_selection",
        "sklearn.utils._typedefs", "sklearn.utils._heap", "sklearn.utils._sorting",
        "sklearn.neighbors._partition_nodes",
        "scipy.special._cdflib", "scipy._lib.messagestream",
        "xgboost", "matplotlib.backends.backend_agg",
    ] + xgb_hidden + sklearn_hidden,
    hookspath=[],
    runtime_hooks=[],
    # Nothing here needs a screen-drawing backend other than the file one, and
    # leaving these out keeps the download a good deal smaller.
    excludes=["PyQt5", "PyQt6", "PySide2", "PySide6", "IPython", "jupyter",
              "notebook", "pytest", "sphinx"],
    noarchive=False,
)
archive = PYZ(analysis.pure)

executable = EXE(
    archive,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="HeartRateVariants",
    debug=False,
    strip=False,
    upx=False,
    console=False,          # no terminal window behind the application
)

bundle = COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    name="HeartRateVariants",
)

# Put the data beside the program, where config.py looks for it when frozen.
target = Path(DISTPATH) / "HeartRateVariants" / "data"
if target.exists():
    shutil.rmtree(target)
shutil.copytree(HERE / "data", target)
