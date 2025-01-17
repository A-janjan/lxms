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

