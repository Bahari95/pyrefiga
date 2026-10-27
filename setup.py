# -*- coding: UTF-8 -*-
#!/usr/bin/env python

"""Keep the optional kernel-compilation command; metadata is in pyproject.toml."""

from pathlib import Path
from setuptools import setup, Command
from importlib.util import find_spec
from importlib.machinery import EXTENSION_SUFFIXES
from shutil import copy2
import subprocess
import sys
from tempfile import TemporaryDirectory

#To pyccelize specific files, enhancing the computation speed for results and mesh adaptation. @bahari
class RunPyccelCommand(Command):
    """Custom command to run pyccel on specific files."""
    description = "Run pyccel on Python files in the specified folder."
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        if find_spec('pyccel') is None:
            raise RuntimeError(
                "Install acceleration support first: "
                "python -m pip install 'pyrefiga[accelerate]'"
            )
        # List of files to process
        files_to_process = [
            'pyrefiga/nurbs_core.py',
            'pyrefiga/nitsche_core.py',
            'pyrefiga/ad_mesh_core.py',
            'pyrefiga/results_f90_core.py',
            'pyrefiga/fast_diag_core.py',
            'pyrefiga/nurbs_utilities_core.py',
            'pyrefiga/adnitsche_core.py',
        ]
        # Use this interpreter and a fresh build directory. Cached native
        # support code from another NumPy version can produce invalid modules.
        entry_point = (
            'from pyccel.commands.console import pyccel_command\n'
            'pyccel_command()'
        )
        with TemporaryDirectory(prefix='pyrefiga-pyccel-') as output:
            for file in files_to_process:
                print(f"Running pyccel compile on {file} with OpenMP...", flush=True)
                subprocess.check_call([
                    sys.executable, '-c', entry_point, 'compile', file,
                    '--openmp', '--output', output,
                ])
                source = Path(file)
                binaries = [Path(output)/(source.stem+suffix) for suffix in EXTENSION_SUFFIXES]
                binary = next((path for path in binaries if path.is_file()), None)
                if binary is None:
                    raise RuntimeError(f'Pyccel did not produce an extension for {file}')
                copy2(binary, source.with_name(binary.name))

        # Create a folder after running pyccel
        folder_name = "newFolder"
        Path(folder_name).mkdir(parents=True, exist_ok=True)
        print(f"Folder '{folder_name}' created successfully.")

def setup_package():
    setup(cmdclass={'run_pyccel': RunPyccelCommand})
if __name__ == "__main__":
    setup_package()

