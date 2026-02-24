"""Named preset configurations for classical shape matchers.

Presets are stored as JSON files in ``benchfum/configs/matchers/``.
Adding a new preset is as simple as adding a new ``.json`` file there.

Usage
-----
>>> from benchfum.presets import MatcherPresets
>>> matcher = MatcherPresets.build("standard")
>>> matcher = MatcherPresets.build("standard", fmap_size=50)   # with override
>>> print(MatcherPresets.list_presets())
>>> print(MatcherPresets.describe("standard"))
"""

import json
from pathlib import Path

from geomfum._build import build_matcher, build_matcher_from_json

_PRESETS_DIR = Path(__file__).parent / "configs" / "matchers"


class MatcherPresets:
    """Named preset configurations for classical matchers.

    Presets are stored as JSON files in ``benchfum/configs/matchers/``.
    Adding a new preset is as simple as adding a new ``.json`` file there.

    Available presets
    -----------------
    - ``quick``        : Small spectrum (k=20), basic ICP. Fast but approximate.
    - ``standard``     : Balanced (k=30), ICP + ZoomOut. Good default.
    - ``precise``      : Larger spectrum (k=50), more iterations. Best quality.
    - ``feature_wks``  : Feature-based matching with WKS. No functional map.

    Examples
    --------
    >>> matcher = MatcherPresets.build("standard")
    >>> matcher = MatcherPresets.build("standard", fmap_size=50)   # override
    >>> print(MatcherPresets.list_presets())
    >>> print(MatcherPresets.describe("standard"))
    """

    @classmethod
    def list_presets(cls):
        """List all available named presets.

        Returns
        -------
        names : list[str]
            Sorted list of preset names.
        """
        if not _PRESETS_DIR.exists():
            return []
        return sorted(f.stem for f in _PRESETS_DIR.glob("*.json"))

    @classmethod
    def build(cls, name, **overrides):
        """Build a matcher from a named preset with optional overrides.

        Parameters
        ----------
        name : str
            Preset name (e.g. ``"standard"``).
        **overrides : dict
            Top-level config keys to override (e.g. ``fmap_size=50``).

        Returns
        -------
        matcher : BaseMatcher
            Configured matcher instance.

        Examples
        --------
        >>> matcher = MatcherPresets.build("standard")
        >>> matcher = MatcherPresets.build("standard", fmap_size=50)
        """
        config_path = _PRESETS_DIR / f"{name}.json"
        if not config_path.exists():
            available = cls.list_presets()
            raise ValueError(f"Unknown preset {name!r}. Available presets: {available}")
        with open(config_path) as f:
            config = json.load(f)
        config.update(overrides)
        return build_matcher(config)

    @classmethod
    def describe(cls, name):
        """Return the raw configuration dict for a named preset.

        Parameters
        ----------
        name : str
            Preset name.

        Returns
        -------
        config : dict
            Raw configuration dictionary (the contents of the JSON file).
        """
        config_path = _PRESETS_DIR / f"{name}.json"
        if not config_path.exists():
            available = cls.list_presets()
            raise ValueError(f"Unknown preset {name!r}. Available presets: {available}")
        with open(config_path) as f:
            return json.load(f)


__all__ = ["MatcherPresets", "build_matcher", "build_matcher_from_json"]
