from src.core.enums import OperandType, ImplType
from src.operators.physical.base import PhysicalOperator
import src.prompts.induce_prompts as prompts
import pandas as pd
from src.utils.parse import list_wrapper

class LLMOnlyInduce(PhysicalOperator):
    def __init__(self, operand_type: OperandType, thinking: bool = False):
        super().__init__(ImplType.LLM_ONLY, operand_type, thinking)
    
    def execute(self, condition: str, df: pd.DataFrame|list, **kwargs) -> str:
        if self.operand_type == OperandType.TABLE:
            return self._table_execute(condition, df, **kwargs)
        elif self.operand_type in [OperandType.CELL, OperandType.COLUMN, OperandType.ROW]:
            return self._execute(condition, df, **kwargs)
        else:
            raise ValueError(f"Unsupported operand type: {self.operand_type}")
        
    def _execute(self, condition: str, df: pd.DataFrame, **kwargs) -> str:
        if self.operand_type == OperandType.COLUMN:
            table = df.to_json()
        else:
            table = df.to_json(orient='records')
        prompt = prompts.qa_prompt.format(table = table, request = condition)
        result = self.client.call([{'role': 'user', 'content': prompt}])
        return result
    
    def _table_execute(self, condition: str, df: list, **kwargs) -> str:
        table_names = kwargs['table_names']

        table_col_prompt = ""
        for (table, table_df) in zip(table_names, df):
            _cols = list_wrapper(table_df.columns)
            _prompt = f"Table '{table}' have columns '{_cols}'"
            table_col_prompt += _prompt + "\n"

        prompt = prompts.table_prompt.format(tables=table_names, 
                                        request=condition, 
                                        table_cols=table_col_prompt)
        result = self.client.call([{'role': 'user', 'content': prompt}])
        return result
