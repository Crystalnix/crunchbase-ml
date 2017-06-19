"""This script contains requests and functions for extracting company information from database."""

import pandas as pd

financial_ipo_offices_products_request = """SELECT company_id, category_code, founded_at, closed_at, country_code, state_code, city, region, average_funded, total_rounds, average_participants, public_at, acquired_at, products_number, offices, acquired_companies 
FROM
	(SELECT * 
	FROM
		(SELECT *
		FROM	
			(SELECT *
			FROM	
				(SELECT *
				FROM	
					(SELECT *
					FROM
						(SELECT id as company_id, category_code, founded_at, closed_at, country_code, state_code, city, region  
						FROM {0}.cb_objects 
						WHERE entity_type='Company') as companies 
					RIGHT JOIN 
						(SELECT object_id as round_company_id, avg(raised_amount_usd) as average_funded, count(*) as total_rounds, avg(participants) as average_participants 
						FROM {0}.cb_funding_rounds
						GROUP BY object_id) as rounds
					ON company_id = round_company_id) as company_rounds
				LEFT JOIN
					(SELECT acquired_object_id as acquired_company_id, acquired_at 
					FROM {0}.cb_acquisitions) as acquisitions
				ON company_id = acquired_company_id) as financial_info
			LEFT JOIN
				(SELECT object_id as ipo_company_id, public_at
				FROM {0}.cb_ipos) as ipo
			ON company_id = ipo_company_id) as financial_ipo_info
		LEFT JOIN
			(SELECT parent_id, count(*) as products_number 
			FROM {0}.cb_objects
			WHERE NOT isnull(parent_id) and entity_type = 'Product'
			GROUP BY parent_id) as products
		ON parent_id = company_id) as financial_ipo_products_info
	LEFT JOIN
		(SELECT object_id as office_company_id, count(*) as offices FROM {0}.cb_offices
		GROUP BY object_id) as offices_info
	ON company_id = office_company_id) as financial_ipo_products_office_info
LEFT JOIN
	(SELECT acquiring_object_id, count(*) as acquired_companies
    FROM {0}.cb_acquisitions
    GROUP by acquiring_object_id) as acquisition_number
ON company_id = acquiring_object_id;"""

degrees_request = """SELECT count(*) as count, company_id, degree_type
                     FROM (SELECT rel.relationship_object_id as company_id, 
                            IF(lower(deg.degree_type) in {1}, lower(deg.degree_type), 'other') as degree_type
                           FROM {0}.cb_relationships as rel 
                           JOIN {0}.cb_degrees as deg 
                           ON rel.person_object_id=deg.object_id) as rel_deg
                     GROUP BY company_id, degree_type;"""

valid_degrees = ('mba', 'phd', 'ms')


def get_financial_ipo_offices_products_query(connection, scheme):
    """
    Returns pandas DataFrame containing company information about finances, ipo, offices and products.
    """

    return pd.read_sql(financial_ipo_offices_products_request.format(scheme, valid_degrees), con=connection)


def get_degrees_query(connection, scheme):
    """
    Returns pandas DataFrame containing information about how much in each company people are with degrees stored in
    valid_degrees.
    """

    return pd.read_sql(degrees_request.format(scheme, valid_degrees), con=connection)