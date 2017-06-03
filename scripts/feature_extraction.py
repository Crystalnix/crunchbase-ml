import pandas as pd
import MySQLdb as sql

user = input("user: ")
password = input("password: ")
db = sql.connect(user=user, passwd=password, db="crunchbase")
companies = pd.read_sql("select id as company_id, name, category_code, status, founded_at, closed_at, country_code, state_code, city, region  from cb_objects where entity_type='Company';", con=db)
acquisitions = pd.read_sql("select acquired_object_id as company_id,  acquired_at from cb_acquisitions;", con=db)
ipos = pd.read_sql("select ipo_id, object_id as company_id, valuation_amount, valuation_currency_code, raised_amount, raised_currency_code, public_at, stock_symbol from cb_ipos;", con=db)
result = companies.merge(ipos, on='company_id', how='left')
result = result.merge(acquisitions, on='company_id', how='left')
result.index = result['company_id']
funding_rounds = pd.read_sql("select funding_round_id, object_id as company_id, funded_at, funding_round_type, funding_round_code, raised_amount_usd, pre_money_valuation_usd, post_money_valuation_usd, participants from cb_funding_rounds;", con=db)
funding_rounds = funding_rounds[funding_rounds['company_id'].isin(result['company_id'])]
round_codes = funding_rounds['funding_round_code'].unique()
funding_round_columns = [column for column in funding_rounds.columns if column not in ['funding_round_id', 'company_id', 'funding_round_code', 'funding_round_type']]
for code in round_codes:
    for column in funding_round_columns:
        result["{0}_{1}".format(code, column)] = 0
funding_groups = funding_rounds.groupby('company_id')
for group_name in funding_groups.groups.keys():
    group = funding_groups.get_group(group_name)
    for row in group.iterrows():
        row = row[1]
        code = row['funding_round_code']
        for column in funding_round_columns:
            result.loc[group_name, "{0}_{1}".format(code, column)] = row[column]

result.to_csv('data/data.csv')
