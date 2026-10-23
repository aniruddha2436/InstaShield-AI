import unittest
from pydantic import ValidationError
from app.schema import Profile
VALID=dict(username='priya.sharma',profile_picture=1,bio_length=90,followers=850,following=600,posts=210,account_age_days=1800,username_changed=0)
class ValidationTests(unittest.TestCase):
    def test_valid(self): self.assertEqual(Profile(**VALID).username,VALID['username'])
    def test_reject_invalid(self):
        for patch in [dict(followers=-1),dict(followers=1.5),dict(followers=True),dict(followers='12'),dict(profile_picture=2),dict(username='@bad'),dict(username=''),dict(username='x'*31),dict(posts=float('nan')),dict(extra=1),dict(account_age_days=100001)]:
            with self.subTest(patch=patch),self.assertRaises(ValidationError): Profile(**{**VALID,**patch})
    def test_missing_required(self):
        data=VALID.copy();data.pop('username_changed')
        with self.assertRaises(ValidationError): Profile(**data)
if __name__=='__main__': unittest.main()
