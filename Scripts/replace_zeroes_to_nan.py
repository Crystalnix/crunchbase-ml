import pandas as pd
import numpy as np

codes = ['b', 'angel', 'a', 'seed', 'c', 'd', 'unattributed', 'debt_round', 'e', 'f', 'private_equity', 'grant', 'post_ipo_equity', 'post_ipo_debt', 'partial', 'convertible', 'crowd', 'g', 'secondary_market', 'crowd_equity']
suff = ['funded_at', 'raised_amount_usd', 'pre_money_valuation_usd', 'post_money_valuation_usd', 'participants']
df = pd.read_csv("data.csv")

for code in codes:
    col = "%s_funded_at" % code
    selection = df[col].apply(float) == 0.0
    df.loc[selection, ["%s_%s" % (code, s) for s in suff]] = np.nan
df.to_csv("data.csv")
