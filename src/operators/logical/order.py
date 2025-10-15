from src.operators.logical.base import LogicalOperator
from src.core.enums import OperandType, ImplType
import src.operators.physical as physical
import pandas as pd 

class LogicalOrder(LogicalOperator):
    def __init__(self, operand_type: OperandType, **kwargs):
        super().__init__(operand_type)
        self.kwargs = kwargs
    
    def execute(self, impl_type: ImplType, condition: str, df: pd.DataFrame, depend_on: str, ascending: bool, k: int, **kwargs):
        thinking = kwargs['thinking'] if 'thinking' in kwargs else False

        if impl_type == ImplType.LLM_ONLY:
            op = physical.LLMOnlyOrder(self.operand_type, thinking=thinking)
        elif impl_type == ImplType.LLM_SEMI:
            self.validate_inputs(['sort_algo'], kwargs)
            op = physical.LLMSemiOrder(self.operand_type, thinking=thinking) # pair-wise LLM compare + sorting algorithms
        elif impl_type == ImplType.LLM_SEMI_OPTIM: 
            op = physical.LLMSemiOptimOrder(self.operand_type, thinking=thinking) # row-wise quantification a score and sort
        else:
            raise ValueError(f"Unsupported impl_type: {impl_type}")
        
        res =  op.execute(condition = condition,
                    df = df.copy(), 
                    depend_on = depend_on,
                    ascending = ascending,
                    k = k,
                    **kwargs)
        _total, _input, _output = op.get_consumed_tokens()
        self.total_tokens += _total
        self.input_tokens += _input
        self.output_tokens += _output
        return res