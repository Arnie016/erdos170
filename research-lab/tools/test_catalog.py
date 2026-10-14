"""Navigation contracts only. These tests make no mathematical claims."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import catalog

class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.data=json.loads((catalog.ROOT/catalog.CATALOG).read_text())
        for p in self.data['problems']:
            paths=[x['path'].split('#')[0] for x in p['resources']]
            for r in p['results']: paths += [r['proof']]+r['code']
            for path in paths:
                dest=self.root/path; dest.parent.mkdir(parents=True,exist_ok=True)
                if Path(path).suffix: dest.write_text('fixture\n')
                else: dest.mkdir(exist_ok=True)
        for m in self.data['maintenance']:
            dest=self.root/m['path'];dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text('maintenance\n')
    def test_valid_catalog(self): catalog.validate(self.data,self.root)
    def test_duplicate_problem_rejected(self):
        self.data['problems'].append(copy.deepcopy(self.data['problems'][0]))
        with self.assertRaises(ValueError): catalog.validate(self.data,self.root)
    def test_duplicate_result_rejected(self):
        p=self.data['problems'][0];p['results'].append(copy.deepcopy(p['results'][0]))
        with self.assertRaises(ValueError): catalog.validate(self.data,self.root)
    def test_false_problem_number_rejected(self):
        self.data['problems'][0]['number']=170
        with self.assertRaises(ValueError): catalog.validate(self.data,self.root)
    def test_related_branch_number_rejected(self):
        self.data['problems'][-1]['number']=999
        with self.assertRaises(ValueError): catalog.validate(self.data,self.root)
    def test_unsupported_solved_label_rejected(self):
        self.data['problems'][0]['status']='SOLVED'
        with self.assertRaises(ValueError): catalog.validate(self.data,self.root)
    def test_lean_claim_without_evidence_rejected(self):
        self.data['problems'][0]['results'][0]['formal_status']='LEAN_VERIFIED_AT_COMMIT'
        with self.assertRaises(ValueError): catalog.validate(self.data,self.root)
    def test_missing_file_rejected(self):
        (self.root/self.data['problems'][0]['results'][0]['proof']).unlink()
        with self.assertRaises(ValueError): catalog.validate(self.data,self.root)
    def test_unindexed_historical_proof_rejected(self):
        p=self.root/'research-lab/studies/unregistered/proofs/result.md';p.parent.mkdir(parents=True);p.write_text('unindexed')
        with self.assertRaises(ValueError): catalog.validate(self.data,self.root)
    def test_path_escape_rejected(self):
        self.data['problems'][0]['results'][0]['proof']='../private.md'
        with self.assertRaises(ValueError): catalog.validate(self.data,self.root)
    def test_missing_evidence_commit_rejected(self):
        self.data['problems'][0]['results'][0]['evidence_commit']='main'
        with self.assertRaises(ValueError): catalog.validate(self.data,self.root)
    def test_generated_pages_deterministic(self):
        self.assertEqual(catalog.render(self.data),catalog.render(copy.deepcopy(self.data)))
        self.assertEqual(len(catalog.render(self.data)),5)
    def test_missing_relative_link_rejected(self):
        with self.assertRaises(ValueError): catalog.check_links(self.root,{'README.md':'[Broken](missing.md)'})
    def test_external_links_not_fetched(self):
        catalog.check_links(self.root,{'README.md':'[Reference](https://example.invalid/)'})

if __name__=='__main__': unittest.main(verbosity=2)
