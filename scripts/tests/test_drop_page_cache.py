"""Unprivileged checks: these tests never install a helper or drop caches."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'scripts/drop-page-cache.sh'


class CacheDropUtility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.helper = Path(cls.directory.name) / 'helper'
        subprocess.run(['cc', '-std=c11', '-Wall', '-Wextra', '-Werror',
                        str(ROOT / 'scripts/native/drop-page-cache.c'),
                        '-o', str(cls.helper)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_help_needs_no_privilege(self):
        for command in ([str(self.helper), '--help'], [str(SCRIPT), '--help']):
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_helper_rejects_other_actions(self):
        for args in (['3'], ['/tmp/control'], ['--help', '3'], ['--install']):
            result = subprocess.run([str(self.helper), *args], capture_output=True)
            self.assertEqual(result.returncode, 64)

    @unittest.skipIf(os.geteuid() == 0, 'No cache-drop execution under root in tests')
    def test_uninstalled_helper_cannot_flush(self):
        result = subprocess.run([str(self.helper)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 77)
        self.assertIn('setuid installation required', result.stderr)

    @unittest.skipIf(os.geteuid() == 0, 'No privileged installation in tests')
    def test_install_entry_rejects_unprivileged_caller(self):
        result = subprocess.run([str(SCRIPT), '--install-helper', str(self.helper), '0'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn('Invalid privileged installation', result.stderr)


if __name__ == '__main__':
    unittest.main()
