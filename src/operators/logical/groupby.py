from src.operators.logical.base import LogicalOperator
from src.core.enums import OperandType, ImplType
import src.operators.physical as physical
import pandas as pd 

class LogicalGroupBy(LogicalOperator):
    def __init__(self, operand_type: OperandType, **kwargs):
        super().__init__(operand_type)
        self.kwargs = kwargs

    # Existing GroupBy only supports scenarios：
    #   1. SELECT dept FROM employees GROUPBY dept; -> The fields to be displayed are the same as those used in the groupby, similar to DISTINCT
    #   2. SELECT agg(salary) FROM employees GROUPBY dept; -> After grouping by the specified columns, apply aggregation functions on other numeric columns
    # Generalized to our scenarios, the two cases correspond to
    #   1. agg = DISTINCT -> return the cluster name
    #   2. agg = COUNT, SUM, MEAN, MIN, MAX ...
        
    def execute(self, impl_type: ImplType, condition: str, df: pd.DataFrame|list, **kwargs):
        if self.operand_type == OperandType.ROW:
            self.validate_inputs(['depend_on'], kwargs)
        if self.operand_type == OperandType.COLUMN:
            self.validate_inputs(['example_num'], kwargs)
        if self.operand_type == OperandType.TABLE:
            self.validate_inputs(['example_num', 'table_names'], kwargs)

        thinking = kwargs['thinking'] if 'thinking' in kwargs else False

        if impl_type == ImplType.LLM_ONLY:
            op = physical.LLMOnlyGroupBy(self.operand_type, thinking=thinking)
        elif impl_type == ImplType.LLM_SEMI:
            op = physical.LLMSemiGroupBy(self.operand_type, thinking=thinking)
        else:
            raise ValueError(f"Unsupported impl_type: {impl_type}")
        
        res = op.execute(condition = condition, 
                          df = df.copy(), 
                          **kwargs)
        _total, _input, _output = op.get_consumed_tokens()
        self.total_tokens += _total
        self.input_tokens += _input
        self.output_tokens += _output
        return res