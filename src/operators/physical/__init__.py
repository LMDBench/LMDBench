from src.operators.physical.base import PhysicalOperator
from src.operators.physical.map import LLMOnlyMap, LLMSemiMap, LLMSemiOptimMap
from src.operators.physical.select import LLMOnlySelect, LLMSemiSelect
from src.operators.physical.impute import LLMOnlyImpute, LLMSemiImpute
from src.operators.physical.groupby import LLMOnlyGroupBy, LLMSemiGroupBy
from src.operators.physical.order import LLMOnlyOrder, LLMSemiOrder, LLMSemiOptimOrder
from src.operators.physical.induce import LLMOnlyInduce