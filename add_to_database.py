import argparse
import os
import shutil
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from utils import get_embedding_function
from langchain.vectorstores.chroma import Chroma


DATA_PATH = "data"
DATABASE_PATH = "chroma"

def main():
    # Check if the database should be cleared (using the --reset flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("Clearing Database")
        clear_database()

    documents = load_documents()
    chunks = chunk_documents(documents)
    add_to_database(chunks)


# add other document types here
# https://python.langchain.com/docs/integrations/document_loaders/
def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def chunk_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex="ln"
    )
    return text_splitter.split_documents(documents)


def add_to_database(chunks: list[Document]):
    db = Chroma(
        persist_directory=DATABASE_PATH,
        embedding_function=get_embedding_function()
    )

    # Only add documents that don't exist in the DB.
    # update to store the hash of the content in the metadata as well, then we can see if the content has changed

    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("No new documents to add")


def calculate_chunk_ids(chunks):
    '''
    Create IDs in form 
    Page Source : Page Number : Chunk Index

    example: "data/monopoly.pdf:6:2"
    
    :param chunks: the chunks to generate an ID for
    '''

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(DATABASE_PATH):
        shutil.rmtree(DATABASE_PATH())


if __name__ == "__main__":
    main()