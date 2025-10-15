from enum import Enum

class OperatorType(Enum):
    MAP = "map"
    SELECT = "select"
    IMPUTE = "impute"
    GROUPBY = "groupby"
    ORDER = "order"
    INDUCE = "induce"
    E2E = "e2e"

class ImplType(Enum):
    LLM_ONLY = "only"
    LLM_SEMI = "semi"
    LLM_SEMI_OPTIM = "semi_prime"
    NL2SQL = "nl2sql"

class OperandType(Enum):
    CELL = "cell"
    ROW = "row"
    COLUMN = "column"
    TABLE = "table"