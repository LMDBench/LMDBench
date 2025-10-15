import pandas as pd 
import os 
import sys
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
sys.path.append(project_root)

from src.operators.logical import LogicalOrder
from src.core.enums import OperandType, ImplType
from src.metrics.classic import f1_score, kendall_tau_at_k

# knowledge
def pipeline_order0(args):
    truth = ['France', 'Spain', 'Germany', 'Poland', 'Italy']
    country = pd.read_csv("./databases/european_football_2/Country.csv")
    op = LogicalOrder(operand_type=OperandType.ROW)
    result = op.execute(impl_type=args.impl,
                           condition="the land area of the country",
                           depend_on= 'name',
                           ascending = False,
                           k = 5,
                           df=country, 
                           sort_algo=args.sort_algo,
                            thinking = args.thinking)
    pred = result['name'].to_list()
    print(pred)
    acc, _, _ = f1_score(truth, pred)
    tau = kendall_tau_at_k(truth, pred)
    return (acc, tau), op.get_tokens()

# knowledge
def pipeline_order1(args):
    truth = ['Germany', 'France', 'Italy', 'England', 'Spain']
    country = pd.read_csv("./databases/european_football_2/Country.csv")
    print(country)
    op = LogicalOrder(operand_type=OperandType.ROW)
    result = op.execute(impl_type=args.impl,
                           condition="the population of the country",
                           depend_on= 'name',
                           ascending = False,
                           k = 5,
                           df=country, 
                           sort_algo=args.sort_algo,
                            thinking = args.thinking)
    pred = result['name'].to_list()
    print(pred)
    acc, _, _ = f1_score(truth, pred)
    tau = kendall_tau_at_k(truth, pred)
    return (acc, tau), op.get_tokens()


# knowledge
def pipeline_order2(args):
    truth = ['London', 'Paris', 'Boston', 'NYC', 'San Francisco'] #['London', 'Paris', 'Boston', 'NYC', 'San Francisco']
    offices = pd.read_csv("./databases/car_retails/offices.csv")
    offices = offices[['officeCode', 'city', 'country']]
    op = LogicalOrder(operand_type=OperandType.ROW)
    result = op.execute(impl_type=args.impl,
                           condition="the latitude of the city",
                           depend_on= 'city',
                           ascending = False,
                           k = 5,
                           df=offices, 
                           sort_algo=args.sort_algo,
                            thinking = args.thinking)
    pred = result['city'].to_list()
    print(pred)
    acc, _, _ = f1_score(truth, pred)
    tau = kendall_tau_at_k(truth, pred)
    return (acc, tau), op.get_tokens()


# knowledge
def pipeline_order3(args):
    truth = ['San Francisco', 'NYC', 'Boston']
    offices = pd.read_csv("./databases/car_retails/offices.csv")
    offices = offices[['officeCode', 'city', 'country']]
    print(offices)
    op = LogicalOrder(operand_type=OperandType.ROW)
    result = op.execute(impl_type=args.impl,
                           condition="the distance from the city to Los Angeles.",
                           depend_on= 'city',
                           ascending = True,
                           k = 3,
                           df=offices, 
                           sort_algo=args.sort_algo,
                            thinking = args.thinking)
    pred = result['city'].to_list()
    print(pred)
    acc, _, _ = f1_score(truth, pred)
    tau = kendall_tau_at_k(truth, pred)
    return (acc, tau), op.get_tokens()

# knowledge
def pipeline_order4(args):
    truth = ['Moana', 'The Princess and the Frog', 'Treasure Planet', 'Hercules']
    movies = pd.read_csv("./databases/disney/director.csv")
    movies = movies[movies['director']=='Ron Clements']
    # print(movies)
    op = LogicalOrder(operand_type=OperandType.ROW)
    result = op.execute(impl_type=args.impl,
                           condition="the duration of the movie listed in the 'name' column",
                           depend_on= 'name',
                           ascending = False,
                           k = 4,
                           df=movies, 
                           sort_algo=args.sort_algo,
                            thinking = args.thinking)
    pred = result['name'].to_list()
    # print(pred)
    acc, _, _ = f1_score(truth, pred)
    tau = kendall_tau_at_k(truth, pred)
    return (acc, tau), op.get_tokens()

# knowledge
def pipeline_order5(args):
    truth = ['101 Dalmatians', 'The Sword in the Stone', 'The Jungle Book', 'The Aristocats']
    movies = pd.read_csv("./databases/disney/director.csv")
    movies = movies[movies['director']=='Wolfgang Reitherman']
    # print(movies)
    op = LogicalOrder(operand_type=OperandType.ROW)
    result = op.execute(impl_type=args.impl,
                           condition="the release year of the movie listed in the 'name' column",
                           depend_on= 'name',
                           ascending = True,
                           k = 4,
                           df=movies, 
                           sort_algo=args.sort_algo,
                            thinking = args.thinking)
    pred = result['name'].to_list()
    # print(pred)
    acc, _, _ = f1_score(truth, pred)
    tau = kendall_tau_at_k(truth, pred)
    return (acc, tau), op.get_tokens()

# fuzzy 
def pipeline_order6(args):
    truth = ['Ashworth', 'Freyre', 'Hernandez', 'Holz']
    customers = pd.read_csv("./databases/car_retails/customers.csv")
    customers = customers[customers['creditLimit'] > 120000]
    customers = customers[['contactLastName', 'contactFirstName']]
    # print(customers)
    op = LogicalOrder(operand_type=OperandType.ROW)
    result = op.execute(impl_type=args.impl,
                           condition="the alphabetical order of the last name listed in the 'contactLastName' column",
                           depend_on= 'contactLastName',
                           ascending = True,
                           k = 4,
                           df=customers, 
                           sort_algo=args.sort_algo,
                            thinking = args.thinking)
    pred = result['contactLastName'].to_list()
    # print(pred)
    acc, _, _ = f1_score(truth, pred)
    tau = kendall_tau_at_k(truth, pred)
    return (acc, tau), op.get_tokens()

# fuzzy 
def pipeline_order7(args):
    truth = ['Piestrzeniewicz', 'Altagar,G M', 'Rodriguez', 'de Castro']
    customers = pd.read_csv("./databases/car_retails/customers.csv")
    customers = customers[customers['creditLimit'] < 3000]
    # print(customers.shape)
    # print(customers['contactLastName'])
    op = LogicalOrder(operand_type=OperandType.ROW)
    result = op.execute(impl_type=args.impl,
                           condition="the string length of the last name listed in the 'contactLastName' column",
                           depend_on= 'contactLastName',
                           ascending = False,
                           k = 4,
                           df=customers, 
                           sort_algo=args.sort_algo,
                            thinking = args.thinking)
    pred = result['contactLastName'].to_list()
    print(pred)
    acc, _, _ = f1_score(truth, pred)
    tau = kendall_tau_at_k(truth, pred)
    return (acc, tau), op.get_tokens()

# reasoning, hard
def pipeline_order8(args):
    truth = ['1998-10-29', '1997-09-30', '1996-09-17', '1994-10-18', '1994-09-01']
    drivers = pd.read_csv("./databases/formula_1/drivers.csv").dropna()
    drivers = drivers[pd.to_datetime(drivers['dob']).dt.year > 1990]
    # print(drivers['dob'])
    op = LogicalOrder(operand_type=OperandType.ROW)
    result = op.execute(impl_type=args.impl,
                           condition="the age of the driver referring to the birthday listed in the 'dob' column",
                           depend_on= 'dob',
                           ascending = True,
                           k = 5,
                           df=drivers, 
                           sort_algo=args.sort_algo,
                            thinking = args.thinking)
    pred = result['dob'].to_list()
    # print(pred)
    acc, _, _ = f1_score(truth, pred)
    tau = kendall_tau_at_k(truth, pred)
    return (acc, tau), op.get_tokens()

# fuzzy, easy
def pipeline_order9(args):
    truth = ["Gintama", "Gintama", "Gintama'", "Gintama': Enchousen", 'Gintama Movie: Kanketsu-hen - Yorozuya yo Eien Nare']
    mylist = pd.read_csv("./databases/anime/my_anime_list.csv")
    mylist = mylist.sort_values(by='Rating', ascending=False)[:10]
    print(mylist)
    op = LogicalOrder(operand_type=OperandType.ROW)
    result = op.execute(impl_type=args.impl,
                           condition="the similarity between the title 'Gintama' and the titles listed in the 'Title' column",
                           depend_on= 'Title',
                           ascending = False,
                           k = 5,
                           df=mylist, 
                           sort_algo=args.sort_algo,
                            thinking = args.thinking)
    pred = result['Title'].to_list()
    print(pred)
    acc, _, _ = f1_score(truth, pred)
    tau = kendall_tau_at_k(truth, pred)
    return (acc, tau), op.get_tokens()


def pipeline_order10(args):
    query = "Rank Top 5 higest rated movies in imdb by their box office from highest to lowest, and return their ID accordingly."
    truth = ['a-1493', 'a-701', 'a-362', 'a-2007', 'a-2148']
    imdb = pd.read_csv("./databases/movies/imdb.csv")
    imdb = imdb.sort_values(by='Rating', ascending=False)[:5]

    op = LogicalOrder(operand_type=OperandType.ROW)
    k = len(imdb)
    result = op.execute(impl_type=args.impl,
                           condition="box office of this movie",
                           depend_on= 'Title',
                           ascending = False,
                           k = k,
                           df=imdb, 
                           sort_algo = args.sort_algo, 
                           thinking = args.thinking)
    pred = result['ID'].tolist()
    acc, _, _ = f1_score(truth, pred)
    tau = kendall_tau_at_k(truth, pred)
    return (acc, tau), op.get_tokens()


def pipeline_order11(args):
    truth = ['Pacific Ocean', 'Atlantic Ocean', 'Indian Ocean', 'Southern Ocean', 'Arctic Ocean']
    sea = pd.read_csv("./databases/mondial_geo/sea.csv")
    op = LogicalOrder(operand_type=OperandType.ROW)
    sea = op.execute(impl_type=args.impl,
                    condition="The area of the sea.",
                    df=sea,
                    depend_on="Name",
                    k=5,
                    ascending=False,
                    sort_algo=args.sort_algo,
                    thinking=args.thinking)
    pred = sea['Name'].tolist()
    print(pred)
    acc, _, _ = f1_score(truth, pred)
    tau = kendall_tau_at_k(truth, pred)
    return (acc, tau), op.get_tokens()


def pipeline_order12(args):
    truth = ['Blank Space', 'Shake It Off', 'Bad Blood']
    music = pd.read_csv("./databases/music/itunes.csv")
    music = music[(music["Artist_Name"] == "Taylor Swift") & (music["Album_Name"] == "1989")]
    op = LogicalOrder(operand_type=OperandType.ROW)
    music = op.execute(impl_type=args.impl,
                    condition="The highest rank and the duration in weeks on the Billboard chart.",
                    df=music,
                    depend_on="Song_Name",
                    k=3,
                    ascending=False,
                    sort_algo=args.sort_algo,
                    thinking=args.thinking)
    pred = music["Song_Name"].tolist()
    print(pred)
    acc, _, _ = f1_score(truth, pred)
    tau = kendall_tau_at_k(truth, pred)
    return (acc, tau), op.get_tokens()


def pipeline_order13(args):
    truth = ['Applause', 'Do What U Want', 'G.U.Y.']
    music = pd.read_csv("./databases/music/itunes.csv")
    music = music[(music["Artist_Name"] == "Lady Gaga") & (music["Album_Name"] == "ARTPOP")]
    op = LogicalOrder(operand_type=OperandType.ROW)
    music = op.execute(impl_type=args.impl,
                    condition="The highest rank and the duration in weeks on the Billboard chart.",
                    df=music,
                    depend_on="Song_Name",
                    k=3,
                    ascending=False,
                    sort_algo=args.sort_algo,
                    thinking=args.thinking)
    pred = music["Song_Name"].tolist()
    print(pred)
    acc, _, _ = f1_score(truth, pred)
    tau = kendall_tau_at_k(truth, pred)
    return (acc, tau), op.get_tokens()


def pipeline_order14(args):
    truth = ['Ships', 'Planes', 'Trains', 'Trucks and Buses', 'Vintage Cars', 'Classic Cars', 'Motorcycles']
    product = pd.read_csv("./databases/car_retails/productlines.csv")
    print(product)
    op = LogicalOrder(operand_type=OperandType.ROW)
    product = op.execute(impl_type=args.impl,
                        condition="The vehicle volume",
                        df=product,
                        depend_on="productLine",
                        k=len(product),
                        ascending=False,
                        sort_algo=args.sort_algo,
                        thinking=args.thinking)
    pred = product["productLine"].tolist()
    print(pred)
    acc, _, _ = f1_score(truth, pred)
    tau = kendall_tau_at_k(truth, pred)
    return (acc, tau), op.get_tokens()


def pipeline_order15(args):
    truth = ['Planes', 'Trains', 'Motorcycles', 'Classic Cars', 'Vintage Cars', 'Trucks and Buses', 'Ships']
    product = pd.read_csv("./databases/car_retails/productlines.csv")
    print(product)
    op = LogicalOrder(operand_type=OperandType.ROW)
    product = op.execute(impl_type=args.impl,
                        condition="The maximum of the vehicle speed",
                        df=product,
                        depend_on="productLine",
                        k=len(product),
                        ascending=False,
                        sort_algo=args.sort_algo,
                        thinking=args.thinking)
    pred = product["productLine"].tolist()
    print(pred)
    acc, _, _ = f1_score(truth, pred)
    tau = kendall_tau_at_k(truth, pred)
    return (acc, tau), op.get_tokens()


def pipeline_order16(args):
    truth = ['English', 'Mandarin', 'French', 'Japanese', 'German', 'Italian']
    language = pd.read_csv("./databases/movie_3/language.csv")
    op = LogicalOrder(operand_type=OperandType.ROW)
    language = op.execute(impl_type=args.impl,
                        condition="The number of language users",
                        df=language,
                        depend_on="name",
                        k=len(language),
                        ascending=False,
                        sort_algo=args.sort_algo,
                        thinking=args.thinking)
    pred = language["name"].tolist()
    print(pred)
    acc, _, _ = f1_score(truth, pred)
    tau = kendall_tau_at_k(truth, pred)
    return (acc, tau), op.get_tokens()


def pipeline_order17(args):
    truth = ['Microsoft', 'Dell', 'HP', 'Lenovo', 'Toshiba', 'Asus', 'Micro-Star International']
    amazon = pd.read_csv("./databases/electronics/amazon.csv")
    amazon = amazon.sort_values(by='Original_Price', ascending=False)[:30]
    amazon = amazon[['Brand']].drop_duplicates()
    print(amazon)
    op = LogicalOrder(operand_type=OperandType.ROW)
    amazon = op.execute(impl_type=args.impl,
                        condition="The market value of the brand",
                        df=amazon,
                        depend_on="Brand",
                        k=len(amazon),
                        ascending=False,
                        sort_algo=args.sort_algo,
                        thinking=args.thinking)
    pred = amazon['Brand'].tolist()
    print(pred)
    acc, _, _ = f1_score(truth, pred)
    tau = kendall_tau_at_k(truth, pred)
    return (acc, tau), op.get_tokens()


def pipeline_order18(args):
    truth = ['Peter Jackson', 'Christopher Nolan', 'Janek Ambros', 'John Cook']
    movies = pd.read_csv("./databases/movies/imdb.csv")
    movies = movies[movies['Rating'] > 8.5]
    movies = movies[['Director']].drop_duplicates()
    print(movies)
    op = LogicalOrder(operand_type=OperandType.ROW)
    movies = op.execute(impl_type=args.impl,
                        condition="The wealth of the director",
                        df=movies,
                        depend_on="Director",
                        k=len(movies),
                        ascending=False,
                        sort_algo=args.sort_algo,
                        thinking=args.thinking)
    pred = movies['Director'].tolist()
    print(pred)
    acc, _, _ = f1_score(truth, pred)
    tau = kendall_tau_at_k(truth, pred)
    return (acc, tau), op.get_tokens()


def pipeline_order19(args):
    truth = ['Indonesia', 'Russia', 'Japan', 'United States', 'China',  'Brazil', 'India', 'Pakistan', 'Nigeria', 'Bangladesh']
    country = pd.read_csv("./databases/mondial_geo/country.csv")
    country = country[country['Population'] > 100000000]
    print(country)
    op = LogicalOrder(operand_type=OperandType.ROW)
    country = op.execute(impl_type=args.impl,
                        condition="The length of the country's coastline",
                        df=country,
                        depend_on="Name",
                        k=len(country),
                        ascending=False,
                        sort_algo=args.sort_algo,
                        thinking=args.thinking)
    pred = country['Name'].tolist()
    print(pred)
    acc, _, _ = f1_score(truth, pred)
    tau = kendall_tau_at_k(truth, pred)
    return (acc, tau), op.get_tokens()


def pipeline_order20(args):
    truth = ['United States', 'China', 'Japan', 'India', 'Russia', 'Brazil', 'Indonesia', 'Nigeria', 'Pakistan', 'Bangladesh']
    country = pd.read_csv("./databases/mondial_geo/country.csv")
    country = country[country['Population'] > 100000000]
    op = LogicalOrder(operand_type=OperandType.ROW)
    country = op.execute(impl_type=args.impl,
                        condition="The 2020 GDP of the country",
                        df=country,
                        depend_on="Name",
                        k=len(country),
                        ascending=False,
                        sort_algo=args.sort_algo,
                        thinking=args.thinking)
    pred = country['Name'].tolist()
    print(pred)
    acc, _, _ = f1_score(truth, pred)
    tau = kendall_tau_at_k(truth, pred)
    return (acc, tau), op.get_tokens()


def pipeline_order21(args):
    truth = ['Toshiba', 'HP', 'Microsoft', 'Lenovo', 'Dell', 'Micro-Star International', 'Asus']
    amazon = pd.read_csv("./databases/electronics/amazon.csv")
    amazon = amazon.sort_values(by='Original_Price', ascending=False)[:30]
    # amazon = amazon[['Brand']].drop_duplicates()
    print(amazon)
    op = LogicalOrder(operand_type=OperandType.ROW)
    amazon = op.execute(impl_type=args.impl,
                        condition="The founding time of the brand",
                        df=amazon,
                        depend_on="Brand",
                        k=len(amazon),
                        ascending=True,
                        sort_algo=args.sort_algo,
                        thinking=args.thinking)
    pred = amazon['Brand'].tolist()
    print(pred)
    acc, _, _ = f1_score(truth, pred)
    tau = kendall_tau_at_k(truth, pred)
    return (acc, tau), op.get_tokens()


def pipeline_order22(args):
    truth = ['Mandarin', 'Japanese', 'German', 'French', 'Italian', 'English']
    language = pd.read_csv("./databases/movie_3/language.csv")
    op = LogicalOrder(operand_type=OperandType.ROW)
    language = op.execute(impl_type=args.impl,
                        condition="The genreral difficulty of the language",
                        df=language,
                        depend_on="name",
                        k=len(language),
                        ascending=False,
                        sort_algo=args.sort_algo,
                        thinking=args.thinking)
    pred = language["name"].tolist()
    print(pred)
    acc, _, _ = f1_score(truth, pred)
    tau = kendall_tau_at_k(truth, pred)
    return (acc, tau), op.get_tokens()


def pipeline_order23(args):
    truth = ['Pacific Ocean', 'Indian Ocean', 'Caribbean Sea']
    sea = pd.read_csv("./databases/mondial_geo/sea.csv")
    op = LogicalOrder(operand_type=OperandType.ROW)
    sea = op.execute(impl_type=args.impl,
                    condition="The species diversity of the sea.",
                    df=sea,
                    depend_on="Name",
                    k=3,
                    ascending=False,
                    sort_algo=args.sort_algo,
                    thinking=args.thinking)
    pred = sea['Name'].tolist()
    print(pred)
    acc, _, _ = f1_score(truth, pred)
    tau = kendall_tau_at_k(truth, pred)
    return (acc, tau), op.get_tokens()


def pipeline_order24(args):
    truth = ['Christopher Nolan', 'Peter Jackson', 'Janek Ambros', 'John Cook']
    movies = pd.read_csv("./databases/movies/imdb.csv")
    movies = movies[movies['Rating'] > 8.5]
    movies = movies[['Director']].drop_duplicates()
    print(movies)
    op = LogicalOrder(operand_type=OperandType.ROW)
    movies = op.execute(impl_type=args.impl,
                        condition="The influence of the director",
                        df=movies,
                        depend_on="Director",
                        k=len(movies),
                        ascending=False,
                        sort_algo=args.sort_algo,
                        thinking=args.thinking)
    pred = movies['Director'].tolist()
    print(pred)
    acc, _, _ = f1_score(truth, pred)
    tau = kendall_tau_at_k(truth, pred)
    return (acc, tau), op.get_tokens()


def pipeline_order25(args):
    truth = ['Born This Way', 'The Edge of Glory', 'Judas']
    music = pd.read_csv("./databases/music/itunes.csv")
    music = music[ (music["Album_Name"] == "Born This Way")]
    print(music)
    op = LogicalOrder(operand_type=OperandType.ROW)
    music = op.execute(impl_type=args.impl,
                    condition="The song popularity.",
                    df=music,
                    depend_on="Song_Name",
                    k=3,
                    ascending=False,
                    sort_algo=args.sort_algo,
                    thinking=args.thinking)
    pred = music["Song_Name"].tolist()
    print(pred)
    acc, _, _ = f1_score(truth, pred)
    tau = kendall_tau_at_k(truth, pred)
    return (acc, tau), op.get_tokens()


def pipeline_order26(args):
    truth = ['Lake Michigan', 'Lake Superior', 'Lake Huron', 'Caspian Sea', 'Lake Victoria', ]
    lake = pd.read_csv("./databases/mondial_geo/lake.csv")
    lake = lake[lake['Area'] > 50000]
    print(lake)
    op = LogicalOrder(operand_type=OperandType.ROW)
    lake = op.execute(impl_type=args.impl,
                    condition="The international tourism popularity of the lake.",
                    df=lake,
                    depend_on="Name",
                    k=len(lake),
                    ascending=False,
                    sort_algo=args.sort_algo,
                    thinking=args.thinking)
    pred = lake["Name"].tolist()
    print(pred)
    acc, _, _ = f1_score(truth, pred)
    tau = kendall_tau_at_k(truth, pred)
    return (acc, tau), op.get_tokens()


def pipeline_order27(args):
    truth = ['statistical-significance', 'normal-distribution', 'hypothesis-testing', 'machine-learning', 
             'classification', 'distributions']
    tags = pd.read_csv("./databases/codebase_community/tags.csv")
    tags = tags[tags['Count'] > 1000]
    op = LogicalOrder(operand_type=OperandType.ROW)
    tags = op.execute(impl_type=args.impl,
                    condition="The string length",
                    df=tags,
                    depend_on="TagName",
                    k=6,
                    ascending=False,
                    sort_algo=args.sort_algo,
                    thinking=args.thinking)
    pred = tags['TagName'].tolist()
    print(pred)
    acc, _, _ = f1_score(truth, pred)
    tau = kendall_tau_at_k(truth, pred)
    return (acc, tau), op.get_tokens()


def pipeline_order28(args):
    truth = ['clustering', 'classification', 'machine-learning']
    tags = pd.read_csv("./databases/codebase_community/tags.csv")
    tags = tags[tags['Count'] > 1000]
    op = LogicalOrder(operand_type=OperandType.ROW)
    tags = op.execute(impl_type=args.impl,
                    condition="the relevance to K-Means",
                    df=tags,
                    depend_on="TagName",
                    k=3,
                    ascending=False,
                    sort_algo=args.sort_algo,
                    thinking=args.thinking)
    pred = tags['TagName'].tolist()
    print(pred)
    acc, _, _ = f1_score(truth, pred)
    tau = kendall_tau_at_k(truth, pred)
    return (acc, tau), op.get_tokens()


def pipeline_order29(args):
    truth = ['time-series', 'regression', 'machine-learning']
    tags = pd.read_csv("./databases/codebase_community/tags.csv")
    tags = tags[tags['Count'] > 1000]
    op = LogicalOrder(operand_type=OperandType.ROW)
    tags = op.execute(impl_type=args.impl,
                    condition="the relevance to time series forecasting",
                    df=tags,
                    depend_on="TagName",
                    k=3,
                    ascending=False,
                    sort_algo=args.sort_algo,
                    thinking=args.thinking)
    pred = tags['TagName'].tolist()
    print(pred)
    acc, _, _ = f1_score(truth, pred)
    tau = kendall_tau_at_k(truth, pred)
    return (acc, tau), op.get_tokens()


if __name__ == '__main__':
    # project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # os.chdir(project_root)
    import argparse
    args = argparse.Namespace(impl=ImplType.LLM_SEMI_OPTIM, thinking=True, sort_algo='simple')
    print(pipeline_order24(args))
    