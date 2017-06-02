import pandas as pd

df = pd.read_csv('../invested_companies_and_degrees_with_offices.csv')
codes = ['b', 'angel', 'a', 'seed', 'c', 'd', 'unattributed', 'debt_round', 'e', 'f', 'private_equity', 'grant', 'post_ipo_equity', 'post_ipo_debt', 'partial', 'convertible', 'crowd', 'g', 'secondary_market', 'crowd_equity']
suffixes = ['funded_at', 'raised_amount_usd', 'pre_money_valuation_usd', 'post_money_valuation_usd', 'participants']
df['average_raised_amount_usd'] = df[['%s_raised_amount_usd' % code for code in codes]].mean(axis=1)
df['average_participants'] = df[['%s_participants' % code for code in codes]].mean(axis=1)
df['total_rounds'] = 0
for code in codes:
    round_info = ['%s_%s' % (code, suff) for suff in suffixes]
    df['total_rounds'] = df['total_rounds'] + df[round_info].notnull().any(axis=1)
    df.drop(round_info, inplace=True, axis=1)
#company_groups = df.groupby('company_id')

df.to_csv('../transformed_data.csv', index=False)
