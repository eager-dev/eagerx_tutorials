import os
import sys
from glob import glob

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors.execute import CellExecutionError


def _notebook_run(path):
    """
    Execute a notebook via nbconvert and collect output.

    Copied from: https://stackoverflow.com/questions/20483313/testing-ipython-notebooks

     :returns (parsed nb object, execution errors)
    """
    kernel_name = "python%d" % sys.version_info[0]
    this_file_directory = os.path.dirname(__file__)
    errors = []

    with open(path) as f:
        nb = nbformat.read(f, as_version=4)
        nb.metadata.get("kernelspec", {})["name"] = kernel_name
        ep = ExecutePreprocessor(kernel_name=kernel_name, timeout=600)  # , allow_errors=True

        try:
            ep.preprocess(nb, {"metadata": {"path": this_file_directory}})

        except CellExecutionError as e:
            if "SKIP" in e.traceback:
                print(str(e.traceback).split("\n")[-2])
            else:
                raise e

    return nb, errors


def test_notebooks():
    return
    # for notebook in glob("./tutorials/**/*.ipynb"):
    #     nb, errors = _notebook_run(notebook)
    #     assert errors == []
