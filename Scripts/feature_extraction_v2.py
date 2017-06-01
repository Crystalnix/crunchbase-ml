import pandas as pd
import MySQLdb as sql

user = input("user: ")
password = input("password: ")
db = sql.connect(user=user, passwd=password, db="crunchbase")
valid_degrees = ['mba', 'phd', 'ms']

degrees = pd.read_sql("""SELECT count(*) as count, company_id, degree_type
                         FROM (SELECT rel.relationship_object_id as company_id, 
                                      IF(lower(deg.degree_type) in ('phd', 'mba', 'ms'), lower(deg.degree_type), 'other') as degree_type
                               FROM crunchbase.cb_relationships as rel join crunchbase.cb_degrees as deg 
                               on rel.person_object_id=deg.object_id) as rel_deg
                         group by company_id, degree_type;""", con=db)

df = pd.read_csv('../invested_companies.csv')
for degree_name in valid_degrees + ['other']:
    deg = degrees[degrees.degree_type == degree_name][['count', 'company_id']]
    deg.columns = ['%s_degree' % degree_name, 'company_id']
    df = df.merge(deg, on='company_id')
df.to_csv('../invested_companies_and_degrees.csv', index=False)


