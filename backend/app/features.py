import re
import pandas as pd

RAW_COLS = ['profile_picture', 'bio_length', 'username', 'followers', 'following', 'posts', 'account_age_days', 'username_changed']
FEATURES = ['profile_picture', 'bio_length', 'followers', 'following', 'posts', 'account_age_days', 'username_changed', 'followers_following_ratio', 'followers_per_day', 'posts_per_day', 'username_digits']

def make_features(raw):
    d = raw.copy()
    d['followers_following_ratio'] = d['followers'] / (d['following'] + 1)
    d['followers_per_day'] = d['followers'] / (d['account_age_days'] + 1)
    d['posts_per_day'] = d['posts'] / (d['account_age_days'] + 1)
    d['username_digits'] = d['username'].astype(str).str.count(r'\d')
    return d[FEATURES]
