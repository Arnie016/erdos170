"""Frozen positive and adversarial controls for the selective verifier."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import verify

class CertificateTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((verify.LAB/'harness/certificates/mod5-field47.json').read_text())
        self.registry = json.loads(verify.REGISTRY.read_text())

    def reject(self, **changes):
        data = copy.deepcopy(self.data)
        data.update(changes)
        with self.assertRaises(ValueError):
            verify.check_character(data)

    def test_positive(self):
        self.assertEqual(verify.check_character(self.data)['exact_order'], 47)

    def test_composite_field(self): self.reject(p=1125)
    def test_identity_root(self): self.reject(r=1)
    def test_wrong_root(self): self.reject(r=44)
    def test_even_order(self): self.reject(order=46)
    def test_wrong_odd_order(self): self.reject(order=45)
    def test_wrong_start(self): self.reject(n0=10)
    def test_wrong_period(self): self.reject(period=22)
    def test_duplicate_holes(self): self.reject(exponents=[0,3,4,5,5])
    def test_polynomial_mutation(self): self.reject(exponents=[0,3,4,5,7])
    def test_bad_hole_count(self): self.reject(offset=4)
    def test_boolean_is_not_integer(self): self.reject(base=True)
    def test_numeric_string_rejected(self): self.reject(p='1129')
    def test_field_budget(self): self.reject(p=65537)
    def test_unknown_certificate_field(self): self.reject(command='echo unsafe')

    def test_registry_unknown_verifier(self):
        self.registry['certificates'][0]['verifier']='shell'
        with self.assertRaises(ValueError): verify.validate_registry(self.registry)

    def test_registry_duplicate_id(self):
        self.registry['certificates'] *= 2
        with self.assertRaises(ValueError): verify.validate_registry(self.registry)

    def test_registry_rejects_command(self):
        self.registry['certificates'][0]['command']='python arbitrary.py'
        with self.assertRaises(ValueError): verify.validate_registry(self.registry)

    def test_false_formal_label(self):
        self.registry['certificates'][0]['formalization']='LEAN_VERIFIED_AT_COMMIT'
        with self.assertRaises(ValueError): verify.validate_registry(self.registry)

    def test_parent_path_rejected(self):
        with self.assertRaises(ValueError): verify.safe_file('../README.md')

    def test_absolute_path_rejected(self):
        with self.assertRaises(ValueError): verify.safe_file('/etc/passwd')

    def test_missing_file(self):
        with self.assertRaises(ValueError): verify.safe_file('harness/absent.json')

    def test_symlink_escape(self):
        with tempfile.TemporaryDirectory() as folder:
            d=Path(folder); lab=d/'lab';lab.mkdir()
            (d/'outside.json').write_text('{}')
            (lab/'link.json').symlink_to(d/'outside.json')
            with self.assertRaises(ValueError): verify.safe_file('link.json', lab)

    def test_unknown_selection_fails_closed(self):
        result=verify.run('not-a-registered-id')
        self.assertEqual(result['status'],'FAIL')
        self.assertEqual(result['records'],[])

    def test_receipt_has_hashes_and_scope(self):
        result=verify.run('mod5-field47')
        self.assertEqual(result['status'],'PASS')
        self.assertEqual(len(result['checker_sha256']),64)
        self.assertEqual(len(result['records']),1)
        self.assertIn('finite_scope',result['records'][0])
        self.assertEqual(result['formalization'],'UNFORMALIZED')

if __name__=='__main__': unittest.main()
