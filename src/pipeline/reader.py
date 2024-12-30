from vtkmodules.vtkIOLegacy import vtkUnstructuredGridReader
from pathlib import Path
from typing import List, Union


class Reader(vtkUnstructuredGridReader):
    def __init__(self, files: Union[Path, List[Path]]):
        super().__init__()
        self._current = 0
        self._files = [files] if isinstance(files, Path) else files

        if isinstance(files, Path):
            self._files = [files]
        else:
            self._files = files

    def next(self):
        self._current += 1
        self.SetFileName(self._files[self._current].as_posix())
        if self._current >= len(self._files):
            raise StopIteration
