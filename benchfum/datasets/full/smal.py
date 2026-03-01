"""SMAL animal body dataset loader.

Layout::

    dataset_dir/
      all0.ply ... all300.ply   <- flat, no subdirectory

All 301 shapes share the same topology, so correspondences are identity.
"""

import os

from geomfum.dataset.torch import ShapeDataset


class SmalDataset(ShapeDataset):
    """SMAL parameterized animal body dataset (301 meshes).

    All shapes are in the dataset root (no ``shapes/`` subdirectory).
    Correspondences default to identity.

    Parameters
    ----------
    dataset_dir : str
        Root directory containing the ``.ply`` files.
    **kwargs
        Forwarded to :class:`~geomfum.dataset.torch.ShapeDataset`.
    """

    def __init__(self, dataset_dir, **kwargs):
        kwargs.setdefault("shapes_subdir", "")
        kwargs.setdefault("correspondences", True)
        super().__init__(dataset_dir, **kwargs)


class SmalrDataset(ShapeDataset):
    """SMAL_r remeshed dataset with explicit subdir/corr conventions.

    Parameters
    ----------
    dataset_dir : str
        Root directory containing ``off/`` and ``corres/``.
    split : str or None
        Optional split name (e.g. ``"train"`` or ``"test"``).  When set,
        only shapes listed in ``{split}.txt`` are loaded (or
        ``{split}_cat.txt`` if ``category=True``).  If ``None``, load all
        shapes found in ``off/``.
    category : bool
        If ``True`` and ``split`` is provided, read ``{split}_cat.txt``
        instead of ``{split}.txt``.  Default ``False``.
    **kwargs
        Forwarded to :class:`~geomfum.dataset.torch.ShapeDataset`.
    """

    def __init__(self, dataset_dir, split=None, category=True, **kwargs):
        kwargs.setdefault("shape_type", "mesh")
        kwargs.setdefault("shapes_subdir", "off")
        kwargs.setdefault("corr_subdir", "corres")
        kwargs.setdefault("correspondences", True)
        kwargs.setdefault("corr_offset", 1)
        kwargs.setdefault("file_extensions", (".off",))

        super().__init__(dataset_dir, **kwargs)

        if split is None:
            return

        split_file = f"{split}_cat.txt" if category else f"{split}.txt"
        split_path = os.path.join(dataset_dir, split_file)
        if not os.path.exists(split_path):
            raise FileNotFoundError(
                f"Split file not found: {split_path!r}. "
                "Expected a file listing shape basenames (one per line)."
            )

        exts = tuple(kwargs.get("file_extensions", (".off",)))
        requested = []
        with open(split_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                name = raw_line.strip()
                if not name:
                    continue

                root, ext = os.path.splitext(name)
                if ext.lower() in exts:
                    shape_file = name
                else:
                    shape_file = f"{name}{exts[0]}"

                requested.append(shape_file)

        missing = [fname for fname in requested if fname not in self.shapes]
        if missing:
            preview = ", ".join(missing[:5])
            if len(missing) > 5:
                preview += ", ..."
            raise FileNotFoundError(
                "Some split-listed shapes were not found in dataset directory: "
                f"{preview}"
            )

        self.shape_files = requested
