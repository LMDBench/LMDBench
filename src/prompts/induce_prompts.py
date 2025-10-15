qa_prompt = '''
Task: You are an expert on answering questions over a relational table. 
You need to generate the corresponding free texts that satisfy the '{request}' request.

Inputs:
- A (part of) relational table
{table}

Instruction:
- Answer the request according to the '{request}' request, based on the values in the table.
- Use external knowledge if necessary.
- If the '{request}' request includes the term 'extract', extract only the substring from the input values that satisfies the '{request}' request as the answer.
- If the '{request}' request includes the term 'summary' or 'summarize', provide a comprehensive and concise summary of the values in the tables, based on the '{request}' request.
- If the '{request}' request includes the term 'describe' or 'description', provide a brief introduction about the information types in this table to help users understand how to use it effectively.
- Otherwise, provide an answer that fully meets the '{request}' request.

Output Format:
Return an answer in free text based on the '{request}' request and the table values. 
You must ensure the answer is comprehensive, high-quality, accurate and fully addressing the '{request}' request; then, make it as concise as possible without compromising these qualities.
'''

table_prompt = '''
Task: You are an expert on answering questions over relational databases. 
You need to generate the corresponding free texts that satisfy the '{request}' request.

Inputs:
- A database with multiple tables '{tables}'
{table_cols}

Instruction:
- Answer the request according to the '{request}' request, based on the values in the table.
- Use external knowledge if necessary.
- If the '{request}' request includes the term 'summary,' provide a comprehensive yet concise summary of the database with the '{tables}' tables, based on the '{request}'.
- If the '{request}' request includes the term 'describe', provide a brief introduction to the types of information contained in the database with the '{tables}' tables, to help users understand how to use this database effectively.
- Otherwise, provide an answer that fully meets the '{request}' request.

Output Format:
Return an answer in free text based on the '{request}' request and the table values. 
You must ensure the answer is comprehensive, high-quality, accurate and fully addressing the '{request}' request; then, make it as concise as possible without compromising these qualities.
'''