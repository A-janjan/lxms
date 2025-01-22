## part 1: general and specialized food information

architecture: **Agentic RAG**

book name: **the new complete book of food**

if answers exists in book return that, **else** use **Tavily API** for internet search.

there is a need for: **LLM filter** for checking the answer, which shouldn't have unrelated context.

And also if the query isn't related to application purpose, it shouldn't be answered.


- use **LlamaParse API** for processing PDF file (book).

- processed text should be divided to chunks via **RecursiveCharacterTextSplitter**. be careful about **chunk_size** and **chunk_overlap** .

- convert **chunks to embeddings** via **sentence_transformers in BAAI/bge-small-en-v1.5** and save them in **LanceDB**.

- use **Hybrid Retrieval** for retrieving information from DB.


## part 2: customer services

- managing orders


three parts:
- order cancelation(    cancel_order(order_id, phone_number)     function)
- tracking order status (     check_order_status(order_id)    function)
- comment to order (      comment_order(order_id, person_name ,comment)      function)

some tips:

- if the operation is successful: show appropriate message

- if the operation has errors: show message

required inputs(throug the chat and natural language):

- order number
- phone number

managing possible scenarios like:

- not entering information
- entering wrong information or incomplete information

in this scenario bot should repeat the request for entering information