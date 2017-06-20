#! /usr/bin/python3
from scripts.feature_extraction import do_feature_extraction
from scripts.model import do_model_building
import argparse

extract_help = """Extract features from database and save them in files located by paths in settings.py. 
Also you should specify database credentials."""
fit_help = 'Build prediction model and print best params and scores.'
parser = argparse.ArgumentParser()
exclusive_group = parser.add_mutually_exclusive_group()
exclusive_group.add_argument('--extract', action='store_true', help=extract_help)
exclusive_group.add_argument('--fit', action='store_true', help=fit_help)
db_group = parser.add_argument_group('Database credentials')
db_group.add_argument('--user', help='database user')
db_group.add_argument('--password', help='database user password')
db_group.add_argument('--scheme', help='database scheme')
args = parser.parse_args()

if args.extract:
    user = args.user
    password = args.password
    scheme = args.scheme
    if user and password and scheme:
        do_feature_extraction(user, password, scheme)
    else:
        parser.error("You should specify all database credentials.")
elif args.fit:
    do_model_building()
else:
    parser.error("You should specify --extract or --fit options.")