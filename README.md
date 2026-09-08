# YouTube Transcript RAG with LangChain, FAISS & OpenAI

A Retrieval-Augmented Generation (RAG) project that fetches an English transcript from a YouTube video, splits it into chunks, creates local embeddings using Hugging Face Sentence Transformers, stores them in FAISS, retrieves relevant transcript sections, and uses an OpenAI chat model to answer questions from the retrieved context.

## Architecture / Workflow

```text
YouTube Video / Video ID
          |
          v
YouTubeTranscriptApi
          |
          v
    Raw Transcript
          |
          v
RecursiveCharacterTextSplitter
 chunk_size=1000
 chunk_overlap=200
          |
          v
    Transcript Chunks
          |
          v
HuggingFaceEmbeddings
all-MiniLM-L6-v2 (LOCAL)
          |
          v
       FAISS
   Vector Store
          |
          v
      Retriever
   similarity, k=4
          |
          v
   Relevant Chunks
          |
          v
    PromptTemplate
 Context + Question
          |
          v
      ChatOpenAI
          |
          v
   StrOutputParser
          |
          v
      Final Answer
```

## RAG Workflow

1. **Indexing** - Fetch the YouTube transcript, split it into chunks, generate embeddings, and store them in FAISS.
2. **Retrieval** - Search FAISS for the four most semantically similar chunks.
3. **Augmentation** - Combine retrieved chunks with the user's question using `PromptTemplate`.
4. **Generation** - Send the augmented prompt to `ChatOpenAI`.
5. **Output parsing** - Convert the LLM response to a string with `StrOutputParser`.

## Technologies

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| YouTube Transcript API | Fetch YouTube captions |
| LangChain | RAG orchestration |
| RecursiveCharacterTextSplitter | Split transcripts |
| Hugging Face Sentence Transformers | Local embeddings |
| FAISS | Vector similarity search |
| OpenAI | Answer generation |
| python-dotenv | Load `.env` variables |
| Jupyter Notebook | Development environment |

## Installation

Create and activate a virtual environment, then install:

```bash
pip install youtube-transcript-api langchain langchain-text-splitters langchain-community langchain-openai langchain-huggingface sentence-transformers faiss-cpu python-dotenv
```

For Jupyter/VS Code notebooks:

```bash
pip install -U jupyter ipywidgets tqdm
```

## Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_CHAT_MODEL=your_chat_model
OPENAI_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### Parameters

`OPENAI_API_KEY` - API key used by `ChatOpenAI`.

`OPENAI_CHAT_MODEL` - OpenAI chat model used for final answer generation. Use a model available to your API account.

`OPENAI_EMBEDDING_MODEL` - The local Hugging Face embedding model. Despite the variable name, the current code does **not** use OpenAI embeddings. Recommended value:

```env
OPENAI_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

A clearer name for a future refactor would be:

```env
HUGGINGFACE_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

Never commit `.env` to Git.

## `.gitignore`

```gitignore
.env
.venv/
__pycache__/
.ipynb_checkpoints/
*.pyc
```

## Project Structure

```text
youtube-info-fetch/
|
+-- .venv/
+-- .env
+-- .gitignore
+-- README.md
+-- youtube-info-fetch.ipynb
```

## Implementation

### 1. Load environment variables

```python
from dotenv import load_dotenv
import os

load_dotenv(override=True)
```

### 2. Fetch YouTube transcript

```python
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

video_id = "Gfr50f6ZBvo"

try:
    yt_api = YouTubeTranscriptApi()

    transcript_data = yt_api.fetch(
        video_id,
        languages=["en"]
    )

    transcript = " ".join(
        item.text for item in transcript_data
    )

    transcript_list = transcript_data

except TranscriptsDisabled:
    print("No captions available for this video")
```

For:

```text
https://www.youtube.com/watch?v=Gfr50f6ZBvo
```

the video ID is:

```text
Gfr50f6ZBvo
```

### 3. Text splitting

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.create_documents([transcript])

print("Number of chunks:", len(chunks))
```

The overlap helps preserve context between adjacent chunks.

### 4. Create the OpenAI LLM

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model=os.getenv("OPENAI_CHAT_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.4,
)
```

The LLM is used for final answer generation.

### 5. Create local embeddings

```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name=os.getenv("OPENAI_EMBEDDING_MODEL")
)
```

Recommended model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The model runs locally, so embedding the transcript does not consume Gemini embedding quota.

### 6. Create FAISS vector store

```python
from langchain_community.vectorstores import FAISS

vector_store = FAISS.from_documents(
    chunks,
    embeddings
)
```

### 7. Create retriever

```python
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)
```

`k=4` means that the four most similar chunks are retrieved.

Test:

```python
docs = retriever.invoke("What is DeepMind?")

for doc in docs:
    print(doc.page_content)
    print("-----")
```

### 8. Prompt / Augmentation

```python
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    template="""
    You are a helpful assistant.

    Answer ONLY from the provided transcript context.

    If the context is insufficient, just say you don't know.

    {context}

    Question: {question}
    """,
    input_variables=["context", "question"]
)
```

Retrieve context:

```python
question = "Is the topic of nuclear fusion discussed in this video? If yes, what was discussed?"

retriever_docs = retriever.invoke(question)

context_text = "\n\n".join(
    doc.page_content
    for doc in retriever_docs
)
```

Create the final prompt:

```python
final_prompt = prompt.invoke({
    "context": context_text,
    "question": question
})
```

### 9. Generate answer

```python
answer = llm.invoke(final_prompt)

print(answer.content)
```

## Building the LangChain Runnable Chain

The project also combines the individual RAG steps into a reusable chain.

### Format documents

```python
def format_docs(retrieved_docs):
    context_text = "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )
    return context_text
```

### RunnableParallel

```python
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)

parallel_chain = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})
```

This creates two paths:

```text
                 User Question
                      |
          +-----------+-----------+
          |                       |
          v                       v
      Retriever          RunnablePassthrough
          |                       |
          v                       v
   Retrieved Docs          Original Question
          |
          v
     format_docs()
          |
          +-----------+-----------+
                      |
                      v
                    Prompt
```

Test:

```python
result = parallel_chain.invoke("Who is Demis")

print(result["question"])
print(result["context"])
```

### Final RAG chain

```python
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

main_chain = (
    parallel_chain
    | prompt
    | llm
    | parser
)
```

Now the entire RAG pipeline can be executed with:

```python
answer = main_chain.invoke("Can you summarize the video")

print(answer)
```

## Complete Chain

```text
Question
   |
   v
RunnableParallel
   |
   +----> Retriever ----> format_docs ----+
   |                                      |
   +----> RunnablePassthrough ------------+
                                          |
                                          v
                                       Prompt
                                          |
                                          v
                                     ChatOpenAI
                                          |
                                          v
                                  StrOutputParser
                                          |
                                          v
                                    Final Answer
```

## Important Design Choice: Local Embeddings

The project originally experimented with Gemini embeddings, but Gemini embedding APIs can have free-tier request limits. The current implementation uses:

```python
HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

Advantages:

- Runs locally.
- No Gemini embedding API quota.
- No embedding API request cost.
- Works directly with FAISS.
- Good lightweight model for learning and prototyping.

The OpenAI API is used separately for the generation step.

```text
Local Hugging Face Embeddings
              +
             FAISS
              +
       OpenAI Chat Model
```

## Common Issues

### `RESOURCE_EXHAUSTED: 429`

If you use Gemini embeddings and receive:

```text
429 RESOURCE_EXHAUSTED
```

you have exceeded the applicable Gemini embedding quota/rate limit.

The current project avoids this by using local Hugging Face embeddings.

### `IProgress not found`

You may see:

```text
TqdmWarning: IProgress not found
```

Install:

```bash
pip install -U jupyter ipywidgets tqdm
```

Then restart the VS Code/Jupyter kernel.

### YouTube captions unavailable

The code catches:

```python
except TranscriptsDisabled:
    print("No captions available for this video")
```

The video needs an accessible English transcript for this implementation.

## Persisting FAISS

For experimentation, `FAISS.from_documents()` can recreate the index each time. For a larger application, save the index:

```python
vector_store.save_local("faiss_index")
```

Later:

```python
vector_store = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)
```

This avoids regenerating embeddings for already-indexed documents.

## Future Improvements

- Accept complete YouTube URLs.
- Support multiple videos.
- Store transcript timestamps as metadata.
- Persist FAISS indexes.
- Add a FastAPI backend.
- Add a React frontend.
- Add streaming responses.
- Add conversation history.
- Add reranking.
- Add hybrid search.
- Add LangSmith tracing/evaluation.
- Replace FAISS with a production vector database when required.
- Dockerize and deploy to AWS, Azure, or GCP.

## Learning Outcomes

This project demonstrates the fundamental RAG concepts:

```text
Data Ingestion
      |
      v
Text Splitting
      |
      v
Embeddings
      |
      v
Vector Store
      |
      v
Retrieval
      |
      v
Context Augmentation
      |
      v
LLM Generation
      |
      v
Final Answer
```

The key RAG idea is that the LLM does not need to receive the entire YouTube transcript. The retriever finds the most relevant transcript chunks and supplies those chunks as context to the LLM.

## License

This project is intended for learning, experimentation, and portfolio development.
