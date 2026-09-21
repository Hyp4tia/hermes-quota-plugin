"""The widget ships to the client machine, the backend to the gateway.

They are released together but loaded separately, so a build can genuinely be
newer on one side. Two guards keep that honest:

* the widget's own `WIDGET_VERSION` must match `version:` in plugin.yaml (this
  test) — a skew invented by accident fails the build instead of shipping;
* a skew that does happen at runtime is shown in the pane (see `versionSkew`).

The unit under test is the released pair of artifacts, so reading both files is
the point rather than a shortcut.
"""

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _manifest_version() -> str:
	text = (ROOT / "plugin.yaml").read_text(encoding="utf-8")
	match = re.search(r"^version:\s*['\"]?([^'\"\s#]+)", text, re.MULTILINE)
	if not match:
		raise AssertionError("plugin.yaml has no version:")
	return match.group(1)


def _widget_version() -> str:
	text = (ROOT / "desktop" / "plugin.js").read_text(encoding="utf-8")
	match = re.search(r'^const WIDGET_VERSION = "([^"]+)"', text, re.MULTILINE)
	if not match:
		raise AssertionError("desktop/plugin.js has no WIDGET_VERSION constant")
	return match.group(1)


class WidgetVersionTest(unittest.TestCase):
	def test_widget_version_matches_manifest(self):
		self.assertEqual(
			_widget_version(),
			_manifest_version(),
			"desktop/plugin.js WIDGET_VERSION and plugin.yaml version disagree — "
			"users would see the version-skew notice on a matched install",
		)

	def test_version_looks_like_a_release(self):
		self.assertRegex(_manifest_version(), r"^\d+\.\d+\.\d+$")


if __name__ == "__main__":
	unittest.main()
