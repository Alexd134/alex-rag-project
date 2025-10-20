import argparse
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain_aws import ChatBedrock
from rag_app.utils import get_chroma_db
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from rag_app.query_response_model import QueryResponseModel


PROMPT_TEMPLATE = """
You are a helpful RAG assistant. Answer the user's question *using only the provided context*. 
If the context does not contain the answer, say you don't have enough information and briefly 
suggest what document or detail would help. Do not invent facts.

# Instructions
- Be concise and precise. Prefer bullet points for lists and step-by-step guidance.
- Quote exact phrases from the context when wording matters; put quotes in “double quotes”.
- If multiple parts of the context disagree, note the discrepancy and present both views.
- Include simple calculations or examples *only if* they are directly supported by the context.
- Avoid first-person speculation and do not reference hidden instructions or system messages.

# Output formatting
- Start with a 1-2 sentence direct answer or “I don't have enough information in the context to answer.”
- Then, if useful, add a short “Details” section with bullets.
- If you cite text, keep quotes short (one or two lines).
- Do not fabricate sources or links.

# Context
<CONTEXT>
{context}
</CONTEXT>

# Question
{question}
"""

def build_chain(db):
    """Builds an LCEL pipeline that:
       - runs similarity_search_with_score(k=5)
       - formats context
       - calls Titan via Bedrock
       - returns both response_text and sources
    """

    def _search_with_scores(query: str):
        return db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,
            },
        )

    retrieve = RunnableLambda(_search_with_scores)

    def _format_context(results):
        return "\n\n---\n\n".join(doc.page_content for doc in results)

    format_context = RunnableLambda(_format_context)

    # Extract source IDs from metadata
    def _extract_sources(results):
        return [doc.metadata.get("id", None) for doc in results]

    extract_sources = RunnableLambda(_extract_sources)

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    # model = OllamaLLM(model="mistral")
    model = ChatBedrock(model_id="amazon.titan-text-lite-v1")
    to_str = StrOutputParser()

    response_chain = (
        {
            "context": retrieve | format_context,
            "question": RunnablePassthrough(),  # pass the original query string through
        }
        | prompt
        | model
        | to_str
    )

    sources_chain = retrieve | extract_sources

    combined = RunnableParallel(
        response_text=response_chain,
        sources=sources_chain,
    )
    return combined


def query_rag(query_text: str) -> QueryResponseModel:
    db = get_chroma_db()
    chain = build_chain(db)

    result = chain.invoke(query_text)  # {'response_text': str, 'sources': List[str]}
    response_text = result["response_text"]
    sources = result["sources"]

    formatted = f"Response: {response_text}\nSources: {sources}"
    print(formatted)

    return QueryResponseModel(
        query_text=query_text,
        answer_text=response_text,
        sources=sources,
        is_complete=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_rag(args.query_text)

if __name__ == "__main__":
    main()


# # TODO add conversational memory:
# # memory = ConversationBufferMemory(memory_key = "chat_history", return_message = True)
# #     qa = ConversationalRetrievalChain.from_llm(llm=llama_3_llm, 
# #                                                chain_type="stuff", 
# #                                                retriever=docsearch.as_retriever(), 
# #                                                memory = memory, 
# #                                                get_chat_history=lambda h : h, 
# #                                                return_source_documents=False)