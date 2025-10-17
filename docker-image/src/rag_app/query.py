import argparse
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain_aws import ChatBedrock
from dataclasses import dataclass
from rag_app.utils import get_chroma_db


PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


@dataclass
class QueryResponse:
    query_text: str
    response_text: str
    sources: list[str]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)

def query_rag(query_text: str):
    db = get_chroma_db()

    # k is the number of top relevant chunks to return
    results = db.similarity_search_with_score(query_text, k=5)
    # TODO could try using something more advanced like RetrievalQA 
    # or use a retriever to get larger chunks after a match and give more context
    # retriever = db.as_retriever(search_type="mmr")
    # results = retriever.invoke(query)
    # add a score threshold and then return a generic message if no good matches

    

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # model = OllamaLLM(model="mistral")
    # model = ChatBedrock(model_id="anthropic.claude-3-haiku-20240307-v1:0")
    model = ChatBedrock(model_id="amazon.titan-text-lite-v1")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)

    return QueryResponse(
        query_text=query_text,
        response_text=response_text,
        sources=sources
    )

# TODO add conversational memory:
# memory = ConversationBufferMemory(memory_key = "chat_history", return_message = True)
#     qa = ConversationalRetrievalChain.from_llm(llm=llama_3_llm, 
#                                                chain_type="stuff", 
#                                                retriever=docsearch.as_retriever(), 
#                                                memory = memory, 
#                                                get_chat_history=lambda h : h, 
#                                                return_source_documents=False)


if __name__ == "__main__":
    main()