from src.operators.logical.base import LogicalOperator
from src.core.enums import OperandType, ImplType
import src.operators.physical as physical
import pandas as pd 


class LogicalInduce(LogicalOperator):
    def __init__(self, operand_type: OperandType, **kwargs):
        super().__init__(operand_type)
        self.kwargs = kwargs
        
    def execute(self, impl_type: ImplType, condition: str, df: pd.DataFrame|list, **kwargs):
        if OperandType == OperandType.TABLE:
            self.validate_inputs(['table_names'], kwargs)

        thinking = kwargs['thinking'] if 'thinking' in kwargs else False

        if impl_type == ImplType.LLM_ONLY:
            op = physical.LLMOnlyInduce(self.operand_type, thinking=thinking)
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
