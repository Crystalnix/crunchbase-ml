import pandas as pd
import MySQLdb as sql

user = input("user: ")
password = input("password: ")
db = sql.connect(user=user, passwd=password, db="crunchbase")
products = pd.read_sql("""SELECT parent_id as company_id, count(*) as products_number FROM crunchbase.cb_objects
                          where not isnull(parent_id) and entity_type = 'Product'
                          group by parent_id;""", con=db)

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

raw_data = pd.read_csv('../data.csv')[['company_id', 'founded_at']]
df = df.merge(raw_data, on='company_id', how='left').drop_duplicates()
is_acquired = df.is_acquired == True
is_not_acquired = ~is_acquired
df.loc[is_acquired, 'age'] = (df[is_acquired]['acquired_at'].apply(pd.to_datetime) - df[is_acquired]['founded_at'].apply(pd.to_datetime))\
                                                                                       .apply(lambda x: x.days)
df.loc[is_not_acquired, 'age'] = (pd.to_datetime('2014-01-01') - df[is_not_acquired]['founded_at'].apply(pd.to_datetime))\
                                                                                       .apply(lambda x: x.days)
ipo_columns = ['valuation_amount', 'valuation_currency_code', 'raised_amount',
              'raised_currency_code', 'public_at', 'stock_symbol']
df.drop(['founded_at', 'acquired_at', 'status'] + ipo_columns, axis=1, inplace=True)
df = df.merge(products, on='company_id', how='left')

df.to_csv('../data/transformed_data.csv', index=False)
