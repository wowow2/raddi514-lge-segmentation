"""Generated reporting preserves output identity and metric denominators."""
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path

from src.layout import write_json
from src.reporting import write_experiment_report


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name)
        write_json(self.repo / "experiments/layout.json", {})
        write_json(self.repo / "config/roots.json", {"profiles": {"local": {
            "experiments": "experiments", "data": "data"}}})
        self.exp = self.repo / "experiments/example"
        write_json(self.exp / "experiment.json", {
            "experiment_id": "example", "outputs": ["final", "best", "missing"],
            "default_output": "best", "model": {"name": "<model>"}})
        for name in ("final", "best", "missing"):
            write_json(self.exp / "outputs" / name / "output.json", {
                "run_id": name, "inference": {"checkpoint": name},
                "data": {"splits_sha256": "original-fold-identity"},
                "paths": {"metrics": f"experiments://example/outputs/{name}/metrics.json"}})
        for name, score in (("best", 0.8), ("final", 0.4)):
            write_json(self.exp / "outputs" / name / "metrics.json", {"results": [
                {"case_id": "Case_P001", "class_id": 3, "dice": score,
                 "gt_present": True, "hd95": 10, "hd95_defined": True},
                {"case_id": "Case_N001", "class_id": 3, "dice": 1,
                 "gt_present": False, "hd95": None, "hd95_defined": False},
                {"case_id": "Case_N001", "class_id": 0, "dice": 0,
                 "hd95": 1000, "hd95_defined": True},
            ]})

    def test_report_includes_all_outputs_and_preserves_metric_semantics(self):
        path = write_experiment_report(self.repo, self.repo / "reports/experiments.html")
        html = path.read_text()
        self.assertIn("1 experiments · 3 output views", html)
        self.assertIn("best · default", html)
        self.assertIn("0.900 ± 0.100 (n=2)", html)
        self.assertIn("0.700 ± 0.300 (n=2)", html)
        self.assertIn("0.800 ± 0.000 (n=1)", html)
        self.assertIn("10.000 ± 0.000 (n=1)", html)
        self.assertIn("Metrics unavailable", html)
        self.assertIn("original-fold-identity", html)
        self.assertIn("&lt;model&gt;", html)
        self.assertNotIn("<model>", html)
        self.assertNotIn("<img", html)
        self.assertIn("EMIDEC dataset", html)
        self.assertIn("not\nblind-test results", html)

    def test_legacy_checkout_is_not_silently_discovered(self):
        (self.repo / "experiments").rename(self.repo / "unregistered")
        with self.assertRaisesRegex(ValueError, "canonical"):
            write_experiment_report(self.repo, self.repo / "report.html")

    def test_only_default_summary_and_detail_are_initially_visible(self):
        class Outputs(HTMLParser):
            def __init__(self):
                super().__init__()
                self.defaults = []
                self.secondary = []
                self.checked = False

            def handle_starttag(self, tag, attributes):
                attrs = dict(attributes)
                classes = attrs.get("class", "").split()
                if "output" in classes:
                    (self.defaults if "default" in classes else self.secondary).append(tag)
                if attrs.get("id") == "defaults-only":
                    self.checked = "checked" in attrs

        path = write_experiment_report(self.repo, self.repo / "report.html")
        html = path.read_text()
        parsed = Outputs()
        parsed.feed(html)
        self.assertTrue(parsed.checked)
        self.assertEqual(parsed.defaults, ["tr", "article"])
        self.assertEqual(parsed.secondary.count("tr"), 2)
        self.assertEqual(parsed.secondary.count("article"), 2)
        self.assertIn("body:not(.show-all) .secondary{display:none}", html)
        self.assertIn("classList.toggle('show-all',!this.checked)", html)


if __name__ == "__main__":
    unittest.main()
