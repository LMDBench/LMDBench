# LLM-Enhanced Database Query Processing Benchmark

Large Language Models (LLMs) have demonstrated significant potential in enhancing database systems, particularly for processing _**beyond-database queries**_ that transcend conventional SQL capabilities through semantic operators. 
However, existing LLM-enhanced query processing frameworks suffer from fragmented operator definitions and inadequate evaluation benchmarks. 
To address these gaps, this work makes three core contributions: First, we establish the _**first unified taxonomy**_ of LLM-enhanced operators---categorizing them into six fundamental types (i.e., <span style="font-family: monospace;">Select</span>, <span style="font-family: monospace;">Match</span>, <span style="font-family: monospace;">Impute</span>, <span style="font-family: monospace;">Cluster</span>, <span style="font-family: monospace;">Order</span>, <span style="font-family: monospace;">Summarize</span>) that comprehensively cover all existing approaches. Second, we introduce **<span style="font-variant: small-caps;">LMDBench</span>**, a novel benchmark featuring 120 end-to-end and 300 operator-level queries spanning 27 databases across 8 domains, designed with multi-dimensional diversity and complete operator coverage to enable holistic system assessment and granular operator analysis. Third, our comprehensive evaluation of eight representative methods in both macro end-to-end and micro operator-level reveals critical insights in four aspects:
(1) performance status of different baselines;
(2) performance relationships w.r.t. a number of design choices (e.g., operator design, LLM selection, prompt design, and implementation strategy);
(3) the principles to improve accuracy and efficiency;
and (4) inherent accuracy-efficiency trade-offs.
These findings establish foundational principles for future LLM-database integration, complemented by proposed research directions for optimized planning and adaptive execution.

## <span style="font-variant: small-caps;">LMDBench</span>
Building upon our comprehensive analysis of LLM-enhanced data processing frameworks, methods, and operators in our paper, 
we introduce a novel benchmark **<span style="font-variant: small-caps;">LMDBench</span>** for holistic assessment of these systems. 
**<span style="font-variant: small-caps;">LMDBench</span>** is natively designed with database objects and LLM-enhanced operators, with multi-dimensional diversity spanning 27 real-world databases across 8 domains and queries varying in complexity. 
**<span style="font-variant: small-caps;">LMDBench</span>** integrates _**120 end-to-end and 300 operator-level**_ queries with balanced difficulty distribution and complete operator coverage across granularities, 
enabling both holistic system assessment and granular operator analysis.

### 1. Database Sources
We collect 27 databases from open-source benchmarks and repositories, with 20 databases from [BIRD](https://bird-bench.github.io/), 5 entity matching databases from the [Magellan Data](https://sites.google.com/site/anhaidgroup/useful-stuff/the-magellan-data-repository), and 2 structured databases from the data lake [NextiaJD](https://www.essi.upc.edu/~snadal/nextiajd.html) and [Santos](https://github.com/northeastern-datalab/santos).

### 2. End-to-End Queries
120 end-to-end queries are depicted in natural language questions to evaluate the end-to-end performance of LLM-enhanced methods in macro scope.
These NL queries and ground truth are provided in `e2e_queries.csv`. 
We also provide handwritten plans of these queries in `eval/e2e_queries.py`.


### 3. Operator-Level Queries
300 operator-level queries are provided as formatted query plans (excluding the planning phase) to test each type of
LLM-enhanced operators in micro scope.
We provide these formatted query plans of six operators respectively in `eval/select_eval.py`, `eval/map_eval.py`, `eval/impute_eval.py`, `eval.groupby_eval.py`, `eval/order_eval.py` and `eval/induce_eval.py`.
Note that ''map'', ''groupby'' and ''induce'' refer to <span style="font-family: monospace;">Match</span>, <span style="font-family: monospace;">Cluster</span> and <span style="font-family: monospace;">Aggregate</span> operators in the paper, respectively. 


## Setup Guide
### 1. Setup Environment
Create a Python 3.10 environment and install the required packages in `requirements.txt`:
```bash
conda create -n lmdbench python=3.10
conda activate lmdbench
pip install -r requirements.txt
```
### 2. Prepare Databases
Download the [databases.zip](https://drive.google.com/uc?export=download&id=1tB2gMT3h92OtzkWr_rHzfJml00fLjiRY) and directly extract it under the project root:
```bash
curl -L 'https://drive.google.com/uc?export=download&id=1tB2gMT3h92OtzkWr_rHzfJml00fLjiRY' -o databases.zip
unzip databases.zip
```

### 3. Configure API Access
Set your Base URL and API key to invoke LLMs:

Linux/MacOS:
```bash
export OPENAI_BASE_URL="https://xxx"
export OPENAI_API_KEY="sk-xxxx"
```
Windows PowerShell:
```shell
$env:OPENAI_BASE_URL="https://xxx"
$env:OPENAI_API_KEY="sk-xxxx"
```
### 4. Evaluation Quickstarts
We provide scripts for quickly running benchmarks. These include both end-to-end evaluations and individual operator-level evaluations:

Run end-to-end evaluation:
```shell
bash ./scripts/e2e_eval.sh
```

Execute individual operator-level evaluations:
```bash
bash ./scripts/select_eval.sh
bash ./scripts/match_eval.sh
bash ./scripts/impute_eval.sh
bash ./scripts/cluster_eval.sh
bash ./scripts/order_eval.sh
```

## Modifying Models and Parameters
### 1. Change Model Settings
Modify LLM configurations (e.g., version, temperature) in `src/conf/conf.json`.

### 2. Customize Operators
Adjust operator options (operands, implementation, CoT, few-shot examples) directly in the code:
```python
from src.operators.logical import LogicalSelect

op = LogicalSelect(operand_type=OperandType.COLUMN)         # Operand type
result = op.execute(impl_type = ImplType.LLM_SEMI,          # Implementation type
                    condition = "The column is related to the SAT test.",
                    df = scores, 
                    example_num = 3,                        # Few-shot examples
                    thinking = True)                        # CoT
```