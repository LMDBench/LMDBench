import argparse
import pandas as pd 
import os 
import sys
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
sys.path.append(project_root)

from src.operators.logical import LogicalGroupBy
from src.core.enums import OperandType, ImplType
from src.metrics.classic import nmi, ari

# knowledge, medium
def pipeline_groupby100(args):
    query = "Group the movies with a rating higher than 8.3 by the continent of nationality of their directors."
    truth = [1, 2, 2, 1, 2, 2, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 3, 1, 2, 2, 1, 4, 1, 5]
    movies = pd.read_csv("./databases/movies/imdb.csv")
    movies = movies[movies['Rating'] > 8.3].drop('Summary', axis=1)
    movies = movies[['Title', 'Director']]
    # print(movies.shape[0])

    op = LogicalGroupBy(operand_type=OperandType.ROW)
    result = op.execute(impl_type=args.impl,
                        condition="Group by the continent of the director's nationality", 
                        df = movies,
                        depend_on = 'Director',
                        thinking = args.thinking)
    print(result)
    pred = result['cluster_label'].tolist()
    print(pred)
    ari_score, nmi_score = ari(truth, pred), nmi(truth, pred)
    return (ari_score, nmi_score), op.get_tokens()


# knowledge, medium
def pipeline_groupby101(args):
    truth = [0, 0, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 3, 2, 1, 1, 0, 1, 
             0, 3, 1, 1, 0, 4, 5, 1, 2, 2, 1, 0, 3, 2, 1, 1, 1, 1, 2, 2, 2, 1, 
             2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 5, 1, 1, 1, 2, 1, 1, 2, 5, 1, 1, 1, 
             0, 2, 1, 1, 2, 1]
    circuits = pd.read_csv("./databases/formula_1/circuits.csv").dropna(subset=['country'])
    circuits = circuits[['circuitId', 'name', 'lat', 'lng']]
    # print(circuits.shape)

    op = LogicalGroupBy(operand_type=OperandType.ROW)
    result = op.execute(impl_type=args.impl,
                        condition="Cluster the data by the continent in which each circuit is located.", 
                        df = circuits,
                        depend_on = 'name',
                        thinking = args.thinking)
    pred = result['cluster_label'].tolist()
    print(pred)
    ari_score, nmi_score = ari(truth, pred), nmi(truth, pred)
    return (ari_score, nmi_score), op.get_tokens()


# knowledge, easy
def pipeline_groupby102(args):
    truth = [0, 0, 0, 1, 0, 0, 0, 2, 0, 1, 3, 4, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 3, 0, 0, 2, 0, 0, 3, 0]
    drivers = pd.read_csv("./databases/formula_1/drivers.csv").dropna()
    drivers = drivers[pd.to_datetime(drivers['dob']).dt.year > 1980]
    # print(drivers.shape)
    op = LogicalGroupBy(operand_type=OperandType.ROW)
    result = op.execute(impl_type=args.impl,
                        condition="Cluster the data by the continent in which the driver's nationaltiy is", 
                        df = drivers,
                        depend_on = 'nationality',
                        thinking = args.thinking)
    pred = result['cluster_label'].tolist()
    print(pred)
    ari_score, nmi_score = ari(truth, pred), nmi(truth, pred)
    return (ari_score, nmi_score), op.get_tokens()


# fuzzy and knowledge, medium
def pipeline_groupby103(args):
    truth = [0, 1, 2, 3, 2, 4, 5, 5, 5, 0, 1, 0, 5, 1, 1, 1, 3, 6, 7, 0, 
             3, 5, 0, 6, 5, 5, 6, 3, 3, 1] 
    amazon = pd.read_csv("./databases/electronics/amazon.csv")[:30]
    amazon = amazon.drop('Brand', axis=1)
    op = LogicalGroupBy(operand_type=OperandType.ROW)
    result = op.execute(impl_type=args.impl,
                        condition="Cluster the data into 'Asus', 'Lenovo', 'HP', 'Dell', 'Acer', 'Toshiba', 'Apple' and 'Other' by their brands", 
                        df = amazon,
                        depend_on = 'Name',
                        thinking = args.thinking)
    pred = result['cluster_label'].tolist()
    print(pred)
    ari_score, nmi_score = ari(truth, pred), nmi(truth, pred)
    return (ari_score, nmi_score), op.get_tokens()

# semantic reasoning, medium
def pipeline_groupby104(args):
    truth = [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0]
    posts = pd.read_csv("./databases/codebase_community/posts.csv")
    posts = posts.sort_values(by='ViewCount', ascending=False)[:20]
    # posts[['Body', 'Tags']].to_csv("tmp.csv")
    op = LogicalGroupBy(operand_type=OperandType.ROW)
    result = op.execute(impl_type=args.impl,
                        condition="Cluster the data into 'statistics' and 'programming & software' 2 clusters by the comment content", 
                        df = posts,
                        depend_on = 'Body',
                        thinking = args.thinking)
    pred = result['cluster_label'].tolist()
    print(pred)
    ari_score, nmi_score = ari(truth, pred), nmi(truth, pred)
    return (ari_score, nmi_score), op.get_tokens()

# semantic reasoning, hard
def pipeline_groupby105(args):
    truth = [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0] # 0 statistics-related, 1 programming-related
    posts = pd.read_csv("./databases/codebase_community/posts.csv")
    posts = posts.sort_values(by='AnswerCount', ascending=False)[:20]
    # posts[['Body', 'Tags']].to_csv("tmp.csv")
    op = LogicalGroupBy(operand_type=OperandType.ROW)
    result = op.execute(impl_type=args.impl,
                        condition="Cluster the data into 'statistics' and 'programming & software' 2 clusters by the theme of the comment content", 
                        df = posts,
                        depend_on = 'Body',
                        thinking = args.thinking)
    pred = result['cluster_label'].tolist()
    print(pred)
    ari_score, nmi_score = ari(truth, pred), nmi(truth, pred)
    return (ari_score, nmi_score), op.get_tokens()

# semantic reasoning, hard
def pipeline_groupby106(args):
    truth = [0, 1, 1, 0, 2, 2, 0, 0, 2, 2, 2, 2, 1, 2, 2, 0, 1, 2, 1, 2] # 0 program-related, 1 ml-related, 2-statistics-related
    posts = pd.read_csv("./databases/codebase_community/posts.csv")
    posts = posts.sort_values(by='FavoriteCount', ascending=False)[:20]
    # posts[['Body', 'Tags']].to_csv("tmp.csv")
    op = LogicalGroupBy(operand_type=OperandType.ROW)
    result = op.execute(impl_type=args.impl,
                        condition="Cluster the data into 'statistics', 'programming & software' and 'machine learning' 3 clusters by the theme of the comment content", 
                        df = posts,
                        depend_on = 'Body',
                        thinking = args.thinking)
    pred = result['cluster_label'].tolist()
    print(pred)
    ari_score, nmi_score = ari(truth, pred), nmi(truth, pred)
    return (ari_score, nmi_score), op.get_tokens()


# fuzzy, easy
def pipeline_groupby107(args):
    truth = [1, 2, 3, 4, 5, 6, 7, 7, 3, 8, 9, 5, 7, 5, 10, 7, 11, 6, 10, 12, 2, 13, 2]
    employees = pd.read_csv("./databases/car_retails/employees.csv")
    op = LogicalGroupBy(operand_type=OperandType.ROW)
    result = op.execute(impl_type=args.impl,
                        condition="Cluster the data by the first letter of the first name of each employee", 
                        df = employees,
                        depend_on = 'firstName',
                        thinking = args.thinking)
    pred = result['cluster_label'].tolist()
    print(pred)
    ari_score, nmi_score = ari(truth, pred), nmi(truth, pred)
    return (ari_score, nmi_score), op.get_tokens()


# fuzzy, easy
def pipeline_groupby108(args):
    truth = [1, 0, 4, 5, 2, 1, 1, 5, 0, 5, 1, 1, 1, 2, 0, 1, 0, 4, 0, 0, 
             1, 3, 0, 0, 4, 4, 0, 1, 0, 0, 0, 0, 0, 4, 0, 0, 4, 0, 0, 4, 
             3, 0, 1, 0, 2, 0, 0, 0, 2, 0]
    products = pd.read_csv("./databases/car_retails/products.csv")[:50]
    op = LogicalGroupBy(operand_type=OperandType.ROW)
    result = op.execute(impl_type=args.impl,
                        condition="Cluster the year by decades", 
                        df = products,
                        depend_on = 'productName',
                        thinking = args.thinking)
    pred = result['cluster_label'].tolist()
    print(pred)
    ari_score, nmi_score = ari(truth, pred), nmi(truth, pred)
    return (ari_score, nmi_score), op.get_tokens()


# fuzzy, medium
def pipeline_groupby109(args):
    truth = [0] + [1]*27 + [0] + [1]*4 + [0]*2 + [1]*2 + [0] + [1]*3 + [0, 1, 0, 1]
    ward = pd.read_csv("./databases/chicago_crime/Ward.csv")
    ward = ward.dropna(subset=['ward_email'])
    # print(ward.shape)
    op = LogicalGroupBy(operand_type=OperandType.ROW)
    result = op.execute(impl_type=args.impl,
                        condition="Cluster the data into 2 clusters by the suffix of the email", 
                        df = ward,
                        depend_on = 'ward_email',
                        thinking = args.thinking)
    pred = result['cluster_label'].tolist()
    print(pred)
    ari_score, nmi_score = ari(truth, pred), nmi(truth, pred)
    return (ari_score, nmi_score), op.get_tokens()


# external knowledge, medium
def pipeline_groupby110(args):
    truth = [1, 2, 1, 1, 1, 2, 1, 1, 3, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1]
    products = pd.read_csv("./databases/car_retails/products.csv")
    productLines = pd.read_csv("./databases/car_retails/productlines.csv")

    productLines = productLines[productLines["productLine"] == "Vintage Cars"]
    merged_df = pd.merge(products, productLines, on="productLine")
    merged_df = merged_df[["productName"]]
    print(len(merged_df))
    op = LogicalGroupBy(operand_type=OperandType.ROW)
    result = op.execute(
        impl_type=args.impl,
        condition="Cluster the product data by their manufacture's country into 'USA', 'Europe', 'Others'.",
        df=merged_df,
        depend_on="productName",
        thinking=args.thinking
    )
    pred = result['cluster_label'].tolist()
    print(pred)
    ari_score, nmi_score = ari(truth, pred), nmi(truth, pred)
    return (ari_score, nmi_score), op.get_tokens()


# external knowledge, medium
def pipeline_groupby111(args):
    truth = [1, 2, 1, 2, 2]
    movies_total_gross = pd.read_csv("./databases/disney/movies_total_gross.csv")
    movies_total_gross = movies_total_gross[['MPAA_rating']].drop_duplicates().dropna()
    print(movies_total_gross)
    op = LogicalGroupBy(operand_type=OperandType.ROW)
    result = op.execute(impl_type=args.impl,
        condition="Group the rating into two levels: 'not suitable for children' and 'suitable for children'.",
        df=movies_total_gross,
        depend_on=["MPAA_rating"],
        thinking=args.thinking,
    )
    pred = result['cluster_label'].tolist()
    print(pred)
    ari_score, nmi_score = ari(truth, pred), nmi(truth, pred)
    return (ari_score, nmi_score), op.get_tokens()


# external knowledge, medium
def pipeline_groupby112(args):
    truth = [1, 1, 2, 1, 1, 2, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1]
    country = pd.read_csv("./databases/mondial_geo/country.csv")
    country = country[country['Population'] > 50000000]
    op = LogicalGroupBy(operand_type=OperandType.ROW)
    result = op.execute(impl_type=args.impl,
                                condition="Group these countries by their economic development level （developed or developing）",
                                df=country,
                                depend_on=["Name"],
                                thinking=args.thinking)
    pred = result['cluster_label'].tolist()
    print(pred)
    ari_score, nmi_score = ari(truth, pred), nmi(truth, pred)
    return (ari_score, nmi_score), op.get_tokens()


# external knowledge, medium
def pipeline_groupby113(args):
    truth = [1, 2, 3, 3, 4, 1, 3, 2, 2, 2, 2, 2, 1, 2, 4, 2, 1, 1, 2, 2, 4, 2, 2]
    brand = pd.read_csv("./databases/beer_factory/rootbeerbrand.csv")
    brand = brand.dropna(subset = ["City", "State", "Country"])
    # print(brand['City'])
    op = LogicalGroupBy(operand_type=OperandType.ROW)
    result = op.execute(impl_type=args.impl,
                                condition="Group by the climate type of the city",
                                df=brand,
                                depend_on=["City", "State", "Country"],
                                thinking=args.thinking)
    pred = result['cluster_label'].tolist()
    print(pred)
    ari_score, nmi_score = ari(truth, pred), nmi(truth, pred)
    return (ari_score, nmi_score), op.get_tokens()

# external knowledge, medium
def pipeline_groupby114(args):
    truth = [1, 2, 3, 1, 4, 1, 2, 1, 2, 5, 1, 1, 6, 3, 1, 1, 1, 3, 1, 2, 1, 3, 3, 3, 1, 3, 1, 1, 1, 1]
    dish = pd.read_csv("./databases/menu/Dish.csv")
    dish = dish.sort_values('highest_price', ascending=False)[:30]
    op = LogicalGroupBy(operand_type=OperandType.ROW)
    result = op.execute(impl_type=args.impl,
                                condition="Group by their cuisine origin country",
                                df=dish,
                                depend_on=["name"],
                                thinking=args.thinking)
    pred = result['cluster_label'].tolist()
    print(pred)
    # print(result[['name', 'cluster_name']])
    ari_score, nmi_score = ari(truth, pred), nmi(truth, pred)
    return (ari_score, nmi_score), op.get_tokens()

# fuzzy, easy
def pipeline_groupby115(args):
    truth = [1, 2, 3, 4, 5, 3, 3, 4, 6, 4, 7, 8, 3, 3, 9, 3, 5, 3, 1, 2, 5, 1, 2, 2, 7, 5, 4, 4, 5, 8]
    customers = pd.read_csv("./databases/beer_factory/customers.csv")
    customers = customers.sort_values('FirstPurchaseDate', ascending=True)[:30]
    op = LogicalGroupBy(operand_type=OperandType.ROW)
    result = op.execute(impl_type=args.impl,
                                condition="Group by their email server",
                                df=customers,
                                depend_on=["Email"],
                                thinking=args.thinking)
    pred = result['cluster_label'].tolist()
    print(pred)
    # print(result[['Email', 'cluster_name']])
    # print(len(result))
    ari_score, nmi_score = ari(truth, pred), nmi(truth, pred)
    return (ari_score, nmi_score), op.get_tokens()

# fuzzy, easy
def pipeline_groupby116(args):
    truth = [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3]
    revenue = pd.read_csv("./databases/disney/revenue.csv")
    op = LogicalGroupBy(operand_type=OperandType.ROW)
    result = op.execute(impl_type=args.impl,
                                condition="Group the years by decades",
                                df=revenue,
                                depend_on=['Year'],
                                thinking=args.thinking)
    pred = result['cluster_label'].tolist()
    print(pred)
    print(result[['Year', 'cluster_name']])
    print(len(result))
    ari_score, nmi_score = ari(truth, pred), nmi(truth, pred)
    return (ari_score, nmi_score), op.get_tokens()

# external knowledge, hard
def pipeline_groupby117(args):
    truth = [1, 2, 2, 2, 3, 3, 3, 1, 3, 3, 3, 4, 4, 3, 3, 3, 3, 3, 5, 1, 3, 3, 3, 5, 3, 3, 3, 3, 3, 1, 1, 2, 3, 3, 4, 2, 3, 3, 3, 3, 3, 3]
    city = pd.read_csv("./databases/olympics/city.csv")
    op = LogicalGroupBy(operand_type=OperandType.ROW)
    result = op.execute(impl_type=args.impl,
                                condition="Group the cities by their climate types",
                                df=city,
                                depend_on=['city_name'],
                                thinking=args.thinking)
    pred = result['cluster_label'].tolist()
    print(pred)
    print(result[['city_name', 'cluster_name']])
    ari_score, nmi_score = ari(truth, pred), nmi(truth, pred)
    return (ari_score, nmi_score), op.get_tokens()

# external knowledge, medium
def pipeline_groupby118(args):
    truth = [1, 1, 1, 1, 2, 1, 1, 2, 2, 1, 2, 3, 2, 1, 1, 4, 1, 4, 5, 1, 2, 1, 1, 2, 1, 4, 1, 1, 1, 3, 1, 1, 2, 1, 4, 2, 1, 4, 1, 2, 1, 1]
    city = pd.read_csv("./databases/olympics/city.csv")
    op = LogicalGroupBy(operand_type=OperandType.ROW)
    result = op.execute(impl_type=args.impl,
                                condition="Group the cities by their continents",
                                df=city,
                                depend_on=['city_name'],
                                thinking=args.thinking)
    pred = result['cluster_label'].tolist()
    print(pred)
    print(result[['city_name', 'cluster_name']])
    ari_score, nmi_score = ari(truth, pred), nmi(truth, pred)
    return (ari_score, nmi_score), op.get_tokens()


# external knowledge, medium
def pipeline_groupby119(args):
    truth = [1, 2, 2, 3, 2, 4, 5, 1]
    sport = pd.read_csv("./databases/olympics/sport.csv")  
    sport = sport[sport['sport_name'].str.startswith('B')]
    op = LogicalGroupBy(operand_type=OperandType.ROW)
    result = op.execute(impl_type=args.impl,
                                condition="Group the sports by their origin country",
                                df=sport,
                                depend_on=['sport_name'],
                                thinking=args.thinking)
    pred = result['cluster_label'].tolist()
    print(pred)
    print(result[['sport_name', 'cluster_name']])
    ari_score, nmi_score = ari(truth, pred), nmi(truth, pred)
    return (ari_score, nmi_score), op.get_tokens()


# external knowledge, medium
def pipeline_groupby120(args):
    truth = [1, 2, 3, 4, 3, 1]
    sport = pd.read_csv("./databases/olympics/sport.csv")  
    sport = sport[sport['sport_name'].str.startswith('T')]
    op = LogicalGroupBy(operand_type=OperandType.ROW)
    result = op.execute(impl_type=args.impl,
                                condition="Group the sports by their origin country",
                                df=sport,
                                depend_on=['sport_name'],
                                thinking=args.thinking)
    pred = result['cluster_label'].tolist()
    print(pred)
    print(result[['sport_name', 'cluster_name']])
    ari_score, nmi_score = ari(truth, pred), nmi(truth, pred)
    return (ari_score, nmi_score), op.get_tokens()


# external knowledge, medium
def pipeline_groupby121(args):
    truth = [1, 1, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 2]
    region = pd.read_csv("./databases/olympics/noc_region.csv")  
    region = region[region['region_name'].str.startswith('C')]
    op = LogicalGroupBy(operand_type=OperandType.ROW)
    result = op.execute(impl_type=args.impl,
                                condition="Group the regions by their economic development level (Developing or Developed)",
                                df=region,
                                depend_on=['region_name'],
                                thinking=args.thinking)
    pred = result['cluster_label'].tolist()
    print(pred)
    print(result[['region_name', 'cluster_name']])
    ari_score, nmi_score = ari(truth, pred), nmi(truth, pred)
    return (ari_score, nmi_score), op.get_tokens()

# semantic, medium
def pipeline_groupby122(args):
    truth = [0, 1, 2, 3, 4, 5, 2, 6, 5, 7, 0, 7, 1, 5, 8, 6, 9, 7, 5, 1, 8, 5, 7, 7, 0, 2, 10, 6, 10, 7]
    film = pd.read_csv("./databases/movie_3/film.csv")
    film = film.sort_values('last_update', ascending=False)[:30]
    op = LogicalGroupBy(operand_type=OperandType.ROW)
    result = op.execute(impl_type=args.impl,
                                condition="Group the films by their types （drama, story, ...）",
                                df=film,
                                depend_on=['title', 'description'],
                                thinking=args.thinking)
    pred = result['cluster_label'].tolist()
    print(pred)
    print(result[['description', 'cluster_name']])
    ari_score, nmi_score = ari(truth, pred), nmi(truth, pred)
    return (ari_score, nmi_score), op.get_tokens()


# external knowledge, medium
def pipeline_groupby123(args):
    truth = [1, 1, 1, 2, 2, 2, 1, 1, 2, 1, 2, 1, 2, 1, 1, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 1, 1, 2, 1, 2, 2, 2, 2]
    city = pd.read_csv("./databases/olympics/city.csv")
    op = LogicalGroupBy(operand_type=OperandType.ROW)
    result = op.execute(impl_type=args.impl,
                                condition="Group the cities into 'Coastal' and 'Inland'",
                                df=city,
                                depend_on=['city_name'],
                                thinking=args.thinking)
    pred = result['cluster_label'].tolist()
    print(pred)
    print(result[['city_name', 'cluster_name']])
    ari_score, nmi_score = ari(truth, pred), nmi(truth, pred)
    return (ari_score, nmi_score), op.get_tokens()


# external knowledge, medium
def pipeline_groupby124(args):
    truth = [1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1]
    lake = pd.read_csv("./databases/mondial_geo/lake.csv")
    lake = lake[lake['Latitude'] > 50]
    print(len(lake))
    op = LogicalGroupBy(operand_type=OperandType.ROW)
    result = op.execute(impl_type=args.impl,
                                condition="Group the lakes by their salinity level",
                                df=lake,
                                depend_on=['Name'],
                                thinking=args.thinking)
    pred = result['cluster_label'].tolist()
    print(pred)
    print(result[['Name', 'cluster_name']])
    ari_score, nmi_score = ari(truth, pred), nmi(truth, pred)
    return (ari_score, nmi_score), op.get_tokens()

# external knowledge, medium
def pipeline_groupby125(args):
    truth = [1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1]
    lake = pd.read_csv("./databases/mondial_geo/lake.csv")
    lake = lake[lake['Latitude'] > 50]
    print(len(lake))
    op = LogicalGroupBy(operand_type=OperandType.ROW)
    result = op.execute(impl_type=args.impl,
                                condition="Group the lakes into 'Endorheic' and 'Exorheic'",
                                df=lake,
                                depend_on=['Name'],
                                thinking=args.thinking)
    pred = result['cluster_label'].tolist()
    print(pred)
    print(result[['Name', 'cluster_name']])
    ari_score, nmi_score = ari(truth, pred), nmi(truth, pred)
    return (ari_score, nmi_score), op.get_tokens()

# reasoning, hard
def pipeline_groupby126(args):
    truth = [1, 2, 3, 3, 4, 5, 6, 6, 6, 3, 6, 1, 5, 5, 1, 7, 1, 1, 6, 2]
    amazon = pd.read_csv("./databases/electronics/amazon.csv")
    amazon = amazon.sort_values('Original_Price', ascending=False)[:20]
    pd.set_option('max_colwidth', None)
    op = LogicalGroupBy(operand_type=OperandType.ROW)
    result = op.execute(impl_type=args.impl,
                                condition="Group by their CPU configuration",
                                df=amazon,
                                depend_on=['Name', 'Features'],
                                thinking=args.thinking)
    pred = result['cluster_label'].tolist()
    print(pred)
    print(result[['Features', 'cluster_name']])
    ari_score, nmi_score = ari(truth, pred), nmi(truth, pred)
    return (ari_score, nmi_score), op.get_tokens()

# reasoning, hard
def pipeline_groupby127(args):
    truth = [1, 2, 1, 1, 1, 3, 1, 1, 3, 4, 1, 1, 1, 1, 1, 5, 1, 3, 1, 3]
    amazon = pd.read_csv("./databases/electronics/amazon.csv")
    amazon = amazon.sort_values('Original_Price', ascending=False)[:20]
    pd.set_option('max_colwidth', None)
    op = LogicalGroupBy(operand_type=OperandType.ROW)
    result = op.execute(impl_type=args.impl,
                                condition="Group by their Memory configuration",
                                df=amazon,
                                depend_on=['Name', 'Features'],
                                thinking=args.thinking)
    pred = result['cluster_label'].tolist()
    print(pred)
    print(result[['Features', 'cluster_name']])
    ari_score, nmi_score = ari(truth, pred), nmi(truth, pred)
    return (ari_score, nmi_score), op.get_tokens()

# external knowledge, hard
def pipeline_groupby128(args):
    truth = [2, 0, 0, 1, 4, 0, 2, 5, 5, 5, 1, 6, 6, 1, 6, 5, 6, 4, 5, 0, 0, 2, 5, 3, 5, 5, 3, 6, 5, 5, 0, 6]
    member = pd.read_csv("./databases/student_club/member.csv")
    major = pd.read_csv("./databases/student_club/major.csv")
    merged = pd.merge(member, major, left_on='link_to_major', right_on='major_id')

    op = LogicalGroupBy(operand_type=OperandType.ROW)
    result = op.execute(impl_type=args.impl,
                                condition="Group their majors into 'Engineering', 'Science', 'Business', 'Language', 'Art', 'Society' and 'Others'",
                                df=merged,
                                depend_on=['major_name'],
                                thinking=args.thinking)
    pred = result['cluster_label'].tolist()
    print(pred)
    print(result[['major_name', 'cluster_name']])
    ari_score, nmi_score = ari(truth, pred), nmi(truth, pred)
    return (ari_score, nmi_score), op.get_tokens()


def pipeline_groupby129(args):
    truth = [1, 1, 2, 2, 3, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 3, 1, 1, 2, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    sea = pd.read_csv("./databases/mondial_geo/sea.csv")
    op = LogicalGroupBy(operand_type=OperandType.ROW)
    result = op.execute(impl_type=args.impl,
                                condition="Group by sea closure (marginal, inland, ...)",
                                df=sea,
                                depend_on=['Name'],
                                thinking=args.thinking)
    pred = result['cluster_label'].tolist()
    print(pred)
    print(result[['Name', 'cluster_name']])
    ari_score, nmi_score = ari(truth, pred), nmi(truth, pred)
    return (ari_score, nmi_score), op.get_tokens()



def pipeline_groupby200(args):
    query = "Group all the columns into time, geographical and other information"
    truth =  [3, 1, 3, 3, 3, 1, 1, 3, 3, 3, 2, 2, 2, 2, 2, 3]
    circuits = pd.read_csv("./databases/formula_1/circuits.csv")
    races = pd.read_csv("./databases/formula_1/races.csv")
    merged_df = pd.merge(races, circuits, on='circuitId')
    # print(merged_df.columns.values)

    op = LogicalGroupBy(operand_type=OperandType.COLUMN)
    result = op.execute(impl_type=args.impl, 
                        condition="Group all the columns into 'time_info', 'geographic_info' and 'other_info'", 
                        df = merged_df,
                        example_num=args.example_num,
                        thinking = args.thinking)
    pred = result['cluster_label'].tolist()
    ari_score, nmi_score = ari(truth, pred), nmi(truth, pred)
    return (ari_score, nmi_score), op.get_tokens()


def pipeline_groupby201(args):
    truth = [0, 0, 0, 4, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
             2, 2, 2, 4, 4, 0, 0, 4, 0, 0, 0, 0, 3, 3, 3, 3, 3, 
             3, 4, 4, 1, 1, 4, 4, 2, 4, 4, 2, 4, 4, 2, 4]
    schools = pd.read_csv("./databases/california_schools/schools.csv")
    op = LogicalGroupBy(operand_type=OperandType.COLUMN)
    result = op.execute(impl_type=args.impl, 
                        condition="Group all the columns into 'id_info', 'location_info', 'contact_info', 'educational_info' and 'other_info'", 
                        df = schools,
                        example_num=args.example_num,
                        thinking = args.thinking)
    pred = result['cluster_label'].tolist()
    ari_score, nmi_score = ari(truth, pred), nmi(truth, pred)
    return (ari_score, nmi_score), op.get_tokens()


def pipeline_groupby202(args):
    truth = [0, 0, 0, 1, 2, 1, 2, 2, 2, 1, 1, 1, 2, 1, 1]
    crime = pd.read_csv("./databases/chicago_crime/Crime.csv")
    op = LogicalGroupBy(operand_type=OperandType.COLUMN)
    result = op.execute(impl_type=args.impl, 
                        condition="Group all the columns into 'basic_info', 'location_info', 'detail_info'", 
                        df = crime,
                        example_num=args.example_num,
                        thinking = args.thinking)
    pred = result['cluster_label'].tolist()
    # print(pred)
    ari_score, nmi_score = ari(truth, pred), nmi(truth, pred)
    return (ari_score, nmi_score), op.get_tokens()


def pipeline_groupby203(args):
    truth = [0, 0, 0, 1, 1, 2, 2, 2, 2]
    transations = pd.read_csv("./databases/debit_card_specializing/transactions_1k.csv")
    op = LogicalGroupBy(operand_type=OperandType.COLUMN)
    result = op.execute(impl_type=args.impl, 
                        condition="Group all the columns into 'order_info', 'user_info', 'product_info'", 
                        df = transations,
                        example_num=args.example_num,
                        thinking = args.thinking)
    pred = result['cluster_label'].tolist()
    # print(pred)
    ari_score, nmi_score = ari(truth, pred), nmi(truth, pred)
    return (ari_score, nmi_score), op.get_tokens()

def pipeline_groupby204(args):
    truth = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5]
    hockey = pd.read_csv("./databases/hockey/Master.csv")
    op = LogicalGroupBy(operand_type=OperandType.COLUMN)
    result = op.execute(impl_type=args.impl, 
                        condition="Group all the columns into 'identification_info', 'physical_info' ,'career_info', 'birth_info', 'death_info'", 
                        df = hockey,
                        example_num=args.example_num,
                        thinking = args.thinking)
    pred = result['cluster_label'].tolist()
    # print(pred)
    ari_score, nmi_score = ari(truth, pred), nmi(truth, pred)
    return (ari_score, nmi_score), op.get_tokens()


# medium
def pipeline_groupby205(args):
    truth = [0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 0]
    customers = pd.read_csv("./databases/beer_factory/customers.csv")
    op = LogicalGroupBy(operand_type=OperandType.COLUMN)
    result = op.execute(impl_type=args.impl, 
                        condition="Group all the columns into 'basic_info', 'contact_info' ,'consuming_info'", 
                        df = customers,
                        example_num=args.example_num,
                        thinking = args.thinking)
    pred = result['cluster_label'].tolist()
    print(pred)
    print(result)
    ari_score, nmi_score = ari(truth, pred), nmi(truth, pred)
    return (ari_score, nmi_score), op.get_tokens()


# medium
def pipeline_groupby206(args):
    truth = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1]
    brand = pd.read_csv("./databases/beer_factory/rootbeerbrand.csv")
    op = LogicalGroupBy(operand_type=OperandType.COLUMN)
    result = op.execute(impl_type=args.impl, 
                        condition="Group all the columns into 'basic_info', 'contact_info' ,'product_info'", 
                        df = brand,
                        example_num=args.example_num,
                        thinking = args.thinking)
    pred = result['cluster_label'].tolist()
    print(pred)
    print(result)
    ari_score, nmi_score = ari(truth, pred), nmi(truth, pred)
    return (ari_score, nmi_score), op.get_tokens()


# hard
def pipeline_groupby207(args):
    truth = [0, 1, 2, 2, 0, 2, 2, 0, 3, 0, 3, 4, 0, 0, 4, 3, 3, 3, 3, 4, 4, 4, 4, 3, 4, 4, 4, 4, 3]
    frpm = pd.read_csv("./databases/california_schools/frpm.csv")
    op = LogicalGroupBy(operand_type=OperandType.COLUMN)
    result = op.execute(impl_type=args.impl, 
                        condition="Group all the columns into 'basic_info', 'geo_info' ,'edu_info', 'time_info', 'economic_info'", 
                        df = frpm,
                        example_num=args.example_num,
                        thinking = args.thinking)
    pred = result['cluster_label'].tolist()
    print(pred)
    print(result)
    ari_score, nmi_score = ari(truth, pred), nmi(truth, pred)
    return (ari_score, nmi_score), op.get_tokens()


# easy
def pipeline_groupby208(args):
    truth = [0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2]
    satscores = pd.read_csv("./databases/california_schools/satscores.csv")
    op = LogicalGroupBy(operand_type=OperandType.COLUMN)
    result = op.execute(impl_type=args.impl, 
                        condition="Group all the columns into 'id_info', 'name_info' ,'edu_info'", 
                        df = satscores,
                        example_num=args.example_num,
                        thinking = args.thinking)
    pred = result['cluster_label'].tolist()
    print(pred)
    print(result)
    ari_score, nmi_score = ari(truth, pred), nmi(truth, pred)
    return (ari_score, nmi_score), op.get_tokens()


# medium
def pipeline_groupby209(args):
    truth = [3, 3, 3, 0, 0, 0, 0, 0, 1, 1, 4, 1, 1, 3, 3, 2, 2, 0, 0]
    community = pd.read_csv("./databases/nextiaJD/community-gardens-and-food-trees.csv")
    op = LogicalGroupBy(operand_type=OperandType.COLUMN)
    result = op.execute(impl_type=args.impl, 
                    condition="Group all the columns into 'geo_info', 'tree_info' ,'contact_info', 'admin_info' and 'other_info'", 
                    df = community,
                    example_num=args.example_num,
                    thinking = args.thinking)
    pred = result['cluster_label'].tolist()
    print(pred)
    print(result)
    ari_score, nmi_score = ari(truth, pred), nmi(truth, pred)
    return (ari_score, nmi_score), op.get_tokens()


# easy
def pipeline_groupby210(args):
    truth = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2]
    ward = pd.read_csv("./databases/chicago_crime/Ward.csv")
    op = LogicalGroupBy(operand_type=OperandType.COLUMN)
    result = op.execute(impl_type=args.impl, 
                    condition="Group all the columns into 'basic_info', 'contact_info' and 'other_info'", 
                    df = ward,
                    example_num=args.example_num,
                    thinking = args.thinking)
    pred = result['cluster_label'].tolist()
    print(pred)
    print(result)
    ari_score, nmi_score = ari(truth, pred), nmi(truth, pred)
    return (ari_score, nmi_score), op.get_tokens()


# medium
def pipeline_groupby211(args):
    truth = [0, 0, 0, 1, 3, 3, 2, 0, 1, 2, 2, 3, 3, 3, 0, 1, 1, 0, 1, 4, 4]
    posts = pd.read_csv("./databases/codebase_community/posts.csv")
    op = LogicalGroupBy(operand_type=OperandType.COLUMN)
    result = op.execute(impl_type=args.impl, 
                    condition="Group all the columns into 'id_info', 'time_info', 'content_info', 'traffic_info' and 'other_info'", 
                    df = posts,
                    example_num=args.example_num,
                    thinking = args.thinking)
    pred = result['cluster_label'].tolist()
    print(pred)
    print(result)
    ari_score, nmi_score = ari(truth, pred), nmi(truth, pred)
    return (ari_score, nmi_score), op.get_tokens()


def pipeline_groupby212(args):
    truth = [1] * 3 + [6] * 3 + [1] + [3] * 2 + [2] * 2 + [4] * 66 + [2] * 8 + [5] * 30
    match = pd.read_csv("./databases/european_football_2/Match.csv")
    op = LogicalGroupBy(operand_type=OperandType.COLUMN)
    result = op.execute(impl_type=args.impl, 
                    condition="Group all the columns into 'player_info', 'score_info', 'team_info', 'id_info' and 'bet_info' and 'other_info'", 
                    df = match,
                    example_num=args.example_num,
                    thinking = args.thinking)
    pred = result['cluster_label'].tolist()
    print(pred)
    pd.set_option('display.max_rows', None)
    print(result)
    ari_score, nmi_score = ari(truth, pred), nmi(truth, pred)
    return (ari_score, nmi_score), op.get_tokens()


def pipeline_groupby213(args):
    truth = [4, 1, 1, 2, 1, 1, 3, 3, 1, 3, 1, 1, 2]
    film = pd.read_csv("./databases/movie_3/film.csv")
    op = LogicalGroupBy(operand_type=OperandType.COLUMN)
    result = op.execute(impl_type=args.impl, 
                    condition="Group all the columns into 'content_info', 'time_info', 'rental_info', and 'other_info'", 
                    df = film,
                    example_num=args.example_num,
                    thinking = args.thinking)
    pred = result['cluster_label'].tolist()
    print(pred)
    print(result)
    ari_score, nmi_score = ari(truth, pred), nmi(truth, pred)
    return (ari_score, nmi_score), op.get_tokens()


def pipeline_groupby214(args):
    truth = [0, 0, 5, 1, 1, 1, 1, 2, 2, 2, 2, 4, 5, 5, 5, 2, 2, 4, 3, 3, 5, 5, 4, 3, 3, 3, 2, 0]
    eo = pd.read_csv("./databases/nextiaJD/eo_pr.csv")
    op = LogicalGroupBy(operand_type=OperandType.COLUMN)
    result = op.execute(impl_type=args.impl, 
                    condition="Group all the columns into 'basic_info', 'location_info', 'organization_info', 'finance_info', 'time_info' and 'other_info'", 
                    df = eo,
                    example_num=args.example_num,
                    thinking = args.thinking)
    pred = result['cluster_label'].tolist()
    print(pred)
    print(result)
    ari_score, nmi_score = ari(truth, pred), nmi(truth, pred)
    return (ari_score, nmi_score), op.get_tokens()

# hard
def pipeline_groupby300(args):
    truth = [2, 2, 1, 2, 1, 2, 1, 2, 1, 3, 1, 3, 1, 3, 3, 3, 2, 1, 1, 3, 2, 3, 1, 1, 3, 1, 2, 2, 2, 2, 2, 3, 3, 2]
    table_names = []
    table_dfs = []
    for filename in os.listdir("./databases/mondial_geo"):
        csv_path = os.path.join("./databases/mondial_geo", filename)
        table_names.append(filename.replace(".csv", ""))
        table_dfs.append(pd.read_csv(csv_path))
    # print(table_names)
    op = LogicalGroupBy(operand_type=OperandType.TABLE)
    result = op.execute(impl_type=args.impl, 
                        condition="Group all the tables into water-related information, land-related information and other information", 
                        df = table_dfs, 
                        table_names = table_names, 
                        example_num=args.example_num,
                        thinking = args.thinking)
    pred = result['cluster_label'].tolist()
    # print(pred)
    ari_score, nmi_score = ari(truth, pred), nmi(truth, pred)
    return (ari_score, nmi_score), op.get_tokens()


def pipeline_groupby301(args):
    truth = [4, 4, 1, 0, 0, 4, 0, 2, 2, 2, 3, 3, 3, 4, 4]
    table_names = []
    table_dfs = []
    for filename in os.listdir("./databases/books"):
        csv_path = os.path.join("./databases/books", filename)
        table_names.append(filename.replace(".csv", ""))
        table_dfs.append(pd.read_csv(csv_path))
    # print(table_names)
    op = LogicalGroupBy(operand_type=OperandType.TABLE)
    result = op.execute(impl_type=args.impl, 
                        condition="Group all the tables into book information, author information, customer information, order information and other information", 
                        df = table_dfs, 
                        table_names = table_names, 
                        example_num=args.example_num,
                        thinking = args.thinking)
    pred = result['cluster_label'].tolist()
    # print(pred)
    ari_score, nmi_score = ari(truth, pred), nmi(truth, pred)
    return (ari_score, nmi_score), op.get_tokens()


def pipeline_groupby302(args):
    truth = [1, 4, 2, 1, 4, 2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 3, 3, 3, 3, 3, 3, 4]
    table_names = []
    table_dfs = []
    for filename in os.listdir("./databases/hockey"):
        csv_path = os.path.join("./databases/hockey", filename)
        table_names.append(filename.replace(".csv", ""))
        table_dfs.append(pd.read_csv(csv_path))
    # print(table_names)
    op = LogicalGroupBy(operand_type=OperandType.TABLE)
    result = op.execute(impl_type=args.impl, 
                        condition="Group all the tables into player info, coach info, team info and other info", 
                        df = table_dfs, 
                        table_names = table_names, 
                        example_num=args.example_num,
                        thinking = args.thinking)
    pred = result['cluster_label'].tolist()
    # print(pred)
    ari_score, nmi_score = ari(truth, pred), nmi(truth, pred)
    return (ari_score, nmi_score), op.get_tokens()


def pipeline_groupby303(args):
    truth = [1, 2, 3, 2, 1, 3, 2, 3, 3, 3, 3, 4, 4, 4, 2, 1]
    table_names = []
    table_dfs = []
    for filename in os.listdir("./databases/movie_3"):
        csv_path = os.path.join("./databases/movie_3", filename)
        table_names.append(filename.replace(".csv", ""))
        table_dfs.append(pd.read_csv(csv_path))
    # print(table_names)
    op = LogicalGroupBy(operand_type=OperandType.TABLE)
    result = op.execute(impl_type=args.impl, 
                        condition="Group all the tables into location info, film info, person-related info and other info", 
                        df = table_dfs, 
                        table_names = table_names, 
                        example_num=args.example_num,
                        thinking = args.thinking)
    pred = result['cluster_label'].tolist()
    # print(pred)
    ari_score, nmi_score = ari(truth, pred), nmi(truth, pred)
    return (ari_score, nmi_score), op.get_tokens()

def pipeline_groupby304(args):
    truth = [1, 2, 2, 3, 2, 3, 2, 2, 2, 2, 2, 4, 4, 4]
    table_names = []
    table_dfs = []
    for filename in os.listdir("./databases/formula_1"):
        csv_path = os.path.join("./databases/formula_1", filename)
        table_names.append(filename.replace(".csv", ""))
        table_dfs.append(pd.read_csv(csv_path))
    # print(table_names)
    op = LogicalGroupBy(operand_type=OperandType.TABLE)
    result = op.execute(impl_type=args.impl, 
                        condition="Group all the tables into location info, person-related info, race-related info and other info", 
                        df = table_dfs, 
                        table_names = table_names, 
                        example_num=args.example_num,
                        thinking = args.thinking)
    pred = result['cluster_label'].tolist()
    # print(pred)
    ari_score, nmi_score = ari(truth, pred), nmi(truth, pred)
    return (ari_score, nmi_score), op.get_tokens()



# medium
def pipeline_groupby305(args):
    truth = [1, 1, 2, 2, 1, 2, 3, 3]
    table_names = []
    table_dfs = []
    for filename in os.listdir("./databases/car_retails"):
        csv_path = os.path.join("./databases/car_retails", filename)
        table_names.append(filename.replace(".csv", ""))
        table_dfs.append(pd.read_csv(csv_path))
    # print(table_names)
    op = LogicalGroupBy(operand_type=OperandType.TABLE)
    result = op.execute(impl_type=args.impl, 
                        condition="Group all the tables into people information, order information and product information", 
                        df = table_dfs, 
                        table_names = table_names, 
                        example_num=args.example_num,
                        thinking = args.thinking)
    pred = result['cluster_label'].tolist()
    print(pred)
    print(result)
    ari_score, nmi_score = ari(truth, pred), nmi(truth, pred)
    return (ari_score, nmi_score), op.get_tokens()

# medium
def pipeline_groupby306(args):
    truth = [1, 4, 4, 2, 2, 1, 3]
    table_names = []
    table_dfs = []
    for filename in os.listdir("./databases/beer_factory"):
        csv_path = os.path.join("./databases/beer_factory", filename)
        table_names.append(filename.replace(".csv", ""))
        table_dfs.append(pd.read_csv(csv_path))
    # print(table_names)
    op = LogicalGroupBy(operand_type=OperandType.TABLE)
    result = op.execute(impl_type=args.impl, 
                        condition="Group all the tables into people info, beer info, order info and other info", 
                        df = table_dfs, 
                        table_names = table_names, 
                        example_num=args.example_num,
                        thinking = args.thinking)
    pred = result['cluster_label'].tolist()
    print(pred)
    print(result)
    ari_score, nmi_score = ari(truth, pred), nmi(truth, pred)
    return (ari_score, nmi_score), op.get_tokens() 


# medium
def pipeline_groupby307(args):
    truth = [1, 1, 3, 2, 2, 3, 3, 1]
    table_names = []
    table_dfs = []
    for filename in os.listdir("./databases/european_football_2"):
        csv_path = os.path.join("./databases/european_football_2", filename)
        table_names.append(filename.replace(".csv", ""))
        table_dfs.append(pd.read_csv(csv_path))
    # print(table_names)
    op = LogicalGroupBy(operand_type=OperandType.TABLE)
    result = op.execute(impl_type=args.impl, 
                        condition="Group all the tables into people info, organizaion info, and other info", 
                        df = table_dfs, 
                        table_names = table_names, 
                        example_num=args.example_num,
                        thinking = args.thinking)
    pred = result['cluster_label'].tolist()
    print(pred)
    print(result)
    ari_score, nmi_score = ari(truth, pred), nmi(truth, pred)
    return (ari_score, nmi_score), op.get_tokens() 

# medium
def pipeline_groupby308(args):
    truth = [1, 2, 2, 2, 2, 3, 1, 2]
    table_names = []
    table_dfs = []
    for filename in os.listdir("./databases/codebase_community"):
        csv_path = os.path.join("./databases/codebase_community", filename)
        table_names.append(filename.replace(".csv", ""))
        table_dfs.append(pd.read_csv(csv_path))
    # print(table_names)
    op = LogicalGroupBy(operand_type=OperandType.TABLE)
    result = op.execute(impl_type=args.impl, 
                        condition="Group all the tables into post info, user info and other info", 
                        df = table_dfs, 
                        table_names = table_names, 
                        example_num=args.example_num,
                        thinking = args.thinking)
    pred = result['cluster_label'].tolist()
    print(pred)
    print(result)
    ari_score, nmi_score = ari(truth, pred), nmi(truth, pred)
    return (ari_score, nmi_score), op.get_tokens() 


# easy
def pipeline_groupby309(args):
    truth = [1, 2, 1, 2, 2, 1, 1]
    table_names = []
    table_dfs = []
    for filename in os.listdir("./databases/chicago_crime"):
        csv_path = os.path.join("./databases/chicago_crime", filename)
        table_names.append(filename.replace(".csv", ""))
        table_dfs.append(pd.read_csv(csv_path))
    # print(table_names)
    op = LogicalGroupBy(operand_type=OperandType.TABLE)
    result = op.execute(impl_type=args.impl, 
                        condition="Group all the tables into location info and case info", 
                        df = table_dfs, 
                        table_names = table_names, 
                        example_num=args.example_num,
                        thinking = args.thinking)
    pred = result['cluster_label'].tolist()
    print(pred)
    print(result)
    ari_score, nmi_score = ari(truth, pred), nmi(truth, pred)
    return (ari_score, nmi_score), op.get_tokens() 



# easy
def pipeline_groupby310(args):
    truth = [1, 2, 1, 2, 2, 1, 1]
    table_names = []
    table_dfs = []
    for filename in os.listdir("./databases/chicago_crime"):
        csv_path = os.path.join("./databases/chicago_crime", filename)
        table_names.append(filename.replace(".csv", ""))
        table_dfs.append(pd.read_csv(csv_path))
    # print(table_names)
    op = LogicalGroupBy(operand_type=OperandType.TABLE)
    result = op.execute(impl_type=args.impl, 
                        condition="Group all the tables into location info and case info", 
                        df = table_dfs, 
                        table_names = table_names, 
                        example_num=args.example_num,
                        thinking = args.thinking)
    pred = result['cluster_label'].tolist()
    print(pred)
    print(result)
    ari_score, nmi_score = ari(truth, pred), nmi(truth, pred)
    return (ari_score, nmi_score), op.get_tokens() 

# hard
def pipeline_groupby311(args):
    truth = [1, 2, 3, 2, 1, 4, 2, 3, 3, 4, 3, 5, 5, 5, 5, 1]
    table_names = []
    table_dfs = []
    for filename in os.listdir("./databases/movie_3"):
        csv_path = os.path.join("./databases/movie_3", filename)
        table_names.append(filename.replace(".csv", ""))
        table_dfs.append(pd.read_csv(csv_path))
    # print(table_names)
    op = LogicalGroupBy(operand_type=OperandType.TABLE)
    result = op.execute(impl_type=args.impl, 
                        condition="Group all the tables into location tables, film tables, people tables, tables for joining and other tables", 
                        df = table_dfs, 
                        table_names = table_names, 
                        example_num=args.example_num,
                        thinking = args.thinking)
    pred = result['cluster_label'].tolist()
    print(pred)
    print(result)
    ari_score, nmi_score = ari(truth, pred), nmi(truth, pred)
    return (ari_score, nmi_score), op.get_tokens() 

# medium
def pipeline_groupby312(args):
    truth = [1, 3, 2, 2, 3, 3, 4, 1, 4, 3, 2]
    table_names = []
    table_dfs = []
    for filename in os.listdir("./databases/olympics"):
        csv_path = os.path.join("./databases/olympics", filename)
        table_names.append(filename.replace(".csv", ""))
        table_dfs.append(pd.read_csv(csv_path))
    # print(table_names)
    op = LogicalGroupBy(operand_type=OperandType.TABLE)
    result = op.execute(impl_type=args.impl, 
                        condition="Group all the tables into location tables, game tables, tables for joining and other tables", 
                        df = table_dfs, 
                        table_names = table_names, 
                        example_num=args.example_num,
                        thinking = args.thinking)
    pred = result['cluster_label'].tolist()
    print(pred)
    print(result)
    ari_score, nmi_score = ari(truth, pred), nmi(truth, pred)
    return (ari_score, nmi_score), op.get_tokens() 

# hard
def pipeline_groupby313(args):
    truth = [1, 1, 2, 2, 2, 2, 3, 3, 3]
    table_names = []
    table_dfs = []
    for filename in os.listdir("./databases/professional_basketball"):
        csv_path = os.path.join("./databases/professional_basketball", filename)
        table_names.append(filename.replace(".csv", ""))
        table_dfs.append(pd.read_csv(csv_path))
    # print(table_names)
    op = LogicalGroupBy(operand_type=OperandType.TABLE)
    result = op.execute(impl_type=args.impl, 
                        condition="Group all the tables into award tables, people tables, match tables and other tables", 
                        df = table_dfs, 
                        table_names = table_names, 
                        example_num=args.example_num,
                        thinking = args.thinking)
    pred = result['cluster_label'].tolist()
    print(pred)
    print(result)
    ari_score, nmi_score = ari(truth, pred), nmi(truth, pred)
    return (ari_score, nmi_score), op.get_tokens() 


def pipeline_groupby314(args):
    truth = [3, 0, 2, 0, 0, 1, 1, 3]
    table_names = []
    table_dfs = []
    for filename in os.listdir("./databases/student_club"):
        csv_path = os.path.join("./databases/student_club", filename)
        table_names.append(filename.replace(".csv", ""))
        table_dfs.append(pd.read_csv(csv_path))
    # print(table_names)
    op = LogicalGroupBy(operand_type=OperandType.TABLE)
    result = op.execute(impl_type=args.impl, 
                        condition="Group all the tables into financal tables, people tables, event tables and other tables", 
                        df = table_dfs, 
                        table_names = table_names, 
                        example_num=args.example_num,
                        thinking = args.thinking)
    pred = result['cluster_label'].tolist()
    print(pred)
    print(result)
    ari_score, nmi_score = ari(truth, pred), nmi(truth, pred)
    return (ari_score, nmi_score), op.get_tokens() 


if __name__ == '__main__':
    args = argparse.Namespace(impl=ImplType.LLM_ONLY, example_num=3, thinking=True)
    print(pipeline_groupby122(args))
    
