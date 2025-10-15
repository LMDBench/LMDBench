import pandas as pd 
import os 
import sys
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
sys.path.append(project_root)

from src.operators.logical import LogicalInduce
from src.core.enums import OperandType, ImplType

# summarize rows
def pipeline_induce0(args):
    query = "Summarize the common characteristics of the songs in the Album 1989 by Taylor Swift."
    itunes = pd.read_csv("./databases/music/itunes.csv")
    itunes = itunes[(itunes['Artist_Name'] == 'Taylor Swift') & (itunes['Album_Name'] == '1989')]
    print(itunes.shape[0])

    op = LogicalInduce(operand_type=OperandType.ROW)
    result = op.execute(ImplType.LLM_ONLY,
                           condition="Summarize the common characteristics of the songs in the Album 1989 by Taylor Swift",
                           df=itunes)
    return result, op.get_tokens()


# summarize rows
def pipeline_induce1(args):
    query = "Summarize the site selection of those circuits in the Southern Hemisphere."
    answer = "-"
    circuits = pd.read_csv("./databases/formula_1/circuits.csv")
    circuits = circuits[circuits['lat'] < 0]


    op = LogicalInduce(operand_type=OperandType.ROW)
    result = op.execute(
        ImplType.LLM_ONLY,
        condition="Summarize the site selection of those circuits in the Southern Hemisphere.",
        df=circuits,
    )
    return result, op.get_tokens()


# summarize rows
def pipeline_induce2(args):
    query = "Summarize the topics about popular tags with Top 10 counts."
    answer = "-"
    tags = pd.read_csv("./databases/codebase_community/tags.csv")
    tags = tags.sort_values(by="Count", ascending=False)[:20]

    op = LogicalInduce(operand_type=OperandType.ROW)
    result = op.execute(
        ImplType.LLM_ONLY,
        condition="Summarize the topics about those popular tags.",
        df=tags,
    )
    return result, op.get_tokens()


# summarize rows
def pipeline_induce3(args):
    query = "Summarize the living country pattern of those customers payed the Top 20 amount money on cars"
    answer = "-"
    customers = pd.read_csv("./databases/car_retails/customers.csv")
    payments = pd.read_csv("./databases/car_retails/payments.csv")
    payments = payments.sort_values(by="amount", ascending=False)[:20]

    merged_df = pd.merge(customers, payments, on="customerNumber")
    merged_df = merged_df[['country']]

    op = LogicalInduce(operand_type=OperandType.ROW)
    result = op.execute(
        ImplType.LLM_ONLY,
        condition="Summarize the living country pattern of those customers payed the Top 20 amount money on cars.",
        df=merged_df,
    )
    return result, op.get_tokens()


# summarize rows
def pipeline_induce4(args):
    movies = pd.read_csv("./databases/movies/imdb.csv")
    movies = movies[movies['Director'] == 'Christopher Nolan']

    op = LogicalInduce(operand_type=OperandType.ROW)
    result = op.execute(
        ImplType.LLM_ONLY,
        condition="Summarize the common characteristics of these movies.",
        df=movies,
    )
    return result, op.get_tokens()



# describe tables
def pipeline_induce5(args):
    query = "What this database 'formula_1' is about?"
    table_names = []
    table_dfs = []
    for filename in os.listdir("./databases/formula_1"):
        csv_path = os.path.join("./databases/formula_1", filename)
        table_names.append(filename.replace(".csv", ""))
        table_dfs.append(pd.read_csv(csv_path))
        
    op = LogicalInduce(operand_type=OperandType.TABLE)
    result = op.execute(ImplType.LLM_ONLY,
                        condition="Describe what this database 'formula_1' is about",
                        df=table_dfs, 
                        table_names = table_names)
    return result, op.get_tokens()

# describe tables
def pipeline_induce6(args):
    query = "What the database 'olympics' is about?"
    table_names = []
    table_dfs = []
    for filename in os.listdir("./databases/olympics"):
        csv_path = os.path.join("./databases/olympics", filename)
        table_names.append(filename.replace(".csv", ""))
        table_dfs.append(pd.read_csv(csv_path))
        
    op = LogicalInduce(operand_type=OperandType.TABLE)
    result = op.execute(ImplType.LLM_ONLY,
                        condition="Describe what this database 'olympics' is about",
                        df=table_dfs, 
                        table_names = table_names)
    return result, op.get_tokens()


# describe multiple tables
def pipeline_induce7(args):
    query = "What the database 'mondial_geo' is about?"
    table_names = []
    table_dfs = []
    for filename in os.listdir("./databases/mondial_geo"):
        csv_path = os.path.join("./databases/mondial_geo", filename)
        table_names.append(filename.replace(".csv", ""))
        table_dfs.append(pd.read_csv(csv_path))
        
    op = LogicalInduce(operand_type=OperandType.TABLE)
    result = op.execute(ImplType.LLM_ONLY,
                        condition="Describe what this database 'mondial_geo' is about",
                        df=table_dfs, 
                        table_names = table_names)
    return result, op.get_tokens()

# describe multiple tables
def pipeline_induce8(args):
    query = "What the database 'beer_factory' is about?"
    table_names = []
    table_dfs = []
    for filename in os.listdir("./databases/beer_factory"):
        csv_path = os.path.join("./databases/beer_factory", filename)
        table_names.append(filename.replace(".csv", ""))
        table_dfs.append(pd.read_csv(csv_path))
        
    op = LogicalInduce(operand_type=OperandType.TABLE)
    result = op.execute(ImplType.LLM_ONLY,
                        condition="Describe what this database 'beer_factory' is about",
                        df=table_dfs, 
                        table_names = table_names)
    return result, op.get_tokens()

# describe multiple tables
def pipeline_induce9(args):
    query = "What the database 'books' is about?"
    table_names = []
    table_dfs = []
    for filename in os.listdir("./databases/books"):
        csv_path = os.path.join("./databases/books", filename)
        table_names.append(filename.replace(".csv", ""))
        table_dfs.append(pd.read_csv(csv_path))
        
    op = LogicalInduce(operand_type=OperandType.TABLE)
    result = op.execute(ImplType.LLM_ONLY,
                        condition="Describe what this database 'books' is about",
                        df=table_dfs, 
                        table_names = table_names)
    return result, op.get_tokens()

if __name__ == '__main__':
    import argparse
    args = argparse.Namespace(impl=ImplType.LLM_ONLY)

    print(pipeline_induce0(args))