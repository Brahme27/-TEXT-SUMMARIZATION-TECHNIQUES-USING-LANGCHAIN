# Text Summarization using LangChain + Groq LLM

This project demonstrates multiple **text summarization techniques** using the `LangChain` framework with **Groq LLM (Gemma2-9B-It)**. It covers summarization methods based on context window availability, ranging from **manual prompting** to **map-reduce and refine chains** for larger documents.


## Prerequisites

* Python
* `langchain`, `langchain-groq`, `python-dotenv`, `langchain_community`, `PyPDFLoader`, etc.
* `.env` file containing your Groq API key:

  ```
  GROQ_API_KEY=your_groq_api_key
  ```

## When Tokens Fit in the LLM Context Window

### 1 Manual Prompting

This method manually creates a `SystemMessage` and `HumanMessage` to provide the LLM with explicit instructions.

```python
chat_message = [
    SystemMessage(content="You are expert with expertise in summarizing speech"),
    HumanMessage(content=f"Please provide a short and concise summary of the following speech:\n speech:{speech}")
]
llm(chat_message)
```

### 2️ Prompt Template with LLMChain

Uses `PromptTemplate` to define reusable prompts, then chains it with the LLM via `LLMChain`.

```python
prompt = PromptTemplate(
    input_variables=["speech", "language"],
    template="""
    Write the summary of the Following Speech:
    Speech:{speech}
    Translate the precise summary to {language}
    """
)

llm_chain = LLMChain(llm=llm, prompt=prompt)
summary = llm_chain.run({'speech': speech, 'language': 'hindi'})
```

## When Tokens Don't Fit in Context Window

### 3 Stuff Document Chain

Loads an entire document and stuffs it into the LLM context window (only works for smaller files).

```python
chain = load_summarize_chain(llm=llm, chain_type="stuff", prompt=prompt)
output_summary = chain.run(docs)
```

Not suitable for large documents, as it may exceed context limits.


### 4️ Map-Reduce Summarization

Breaks the document into chunks → summarizes each chunk → combines all summaries into a final output.

```python
summary_chain = load_summarize_chain(
    llm=llm,
    chain_type="map_reduce",
    map_prompt=map_prompt_template,
    combine_prompt=final_prompt_template
)

final_summary = summary_chain.run(chunk_docs)
```

### 5️ Refine Chain Summarization

Initial summary is generated → refined with each new chunk to improve the result iteratively.

```python
refine_chain = load_summarize_chain(llm=llm, chain_type="refine")
output_summary = refine_chain.run(chunk_docs)
```


## Output Example

* Concise summary
* Translated summary
* Bullet-point highlights
* Motivational title

##  Future Enhancements

* Add streaming summarization