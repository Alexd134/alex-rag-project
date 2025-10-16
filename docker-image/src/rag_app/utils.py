from langchain_aws.embeddings.bedrock import BedrockEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import os
import sys
import shutil

CHROMA_DB_INSTANCE = None  # Reference to singleton instance of ChromaDB
CHROMA_PATH = os.environ.get("CHROMA_PATH", "src/data/chroma")
IS_USING_IMAGE_RUNTIME = bool(os.environ.get("IS_USING_IMAGE_RUNTIME", False))


# def get_embedding_function():
#     embeddings = BedrockEmbeddings(credentials_profile_name="default")
#     # embeddings = OllamaEmbeddings(model="nomic-embed-text")
#     return embeddings

def get_embedding_function():
    """Get embedding function using AWS Bedrock with configured credentials"""
    region = os.environ.get("AWS_DEFAULT_REGION", "eu-west-2")

    try:
        embeddings = BedrockEmbeddings(
            model_id="cohere.embed-english-v3",
            region_name=region
        )
        # Test the connection
        embeddings.embed_query("test")
        print("Successfully connected to AWS Bedrock embeddings")
        return embeddings
    except Exception as e:
        print(f"AWS Bedrock embeddings error: {str(e)}")
        print("Check:")
        print("   1. Your AWS credentials are valid")
        print("   2. Your configured region has access to Bedrock")
        print("   3. Your AWS account has Bedrock enabled")
        raise e


def get_chroma_db():
    global CHROMA_DB_INSTANCE
    if not CHROMA_DB_INSTANCE:
        # Hack needed for AWS Lambda's base Python image (to work with an updated version of SQLite).
        # In Lambda runtime, we need to copy ChromaDB to /tmp so it can have write permissions.
        if IS_USING_IMAGE_RUNTIME:
            # __import__("pysqlite3")
            # sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
            copy_chroma_to_tmp()

        # Prepare the DB.
        CHROMA_DB_INSTANCE = Chroma(
            persist_directory=get_runtime_chroma_path(),
            embedding_function=get_embedding_function(),
        )

        print(f"Init ChromaDB {CHROMA_DB_INSTANCE} from {get_runtime_chroma_path()}")

    return CHROMA_DB_INSTANCE

def copy_chroma_to_tmp():
    dst_chroma_path = get_runtime_chroma_path()

    if not os.path.exists(dst_chroma_path):
        os.makedirs(dst_chroma_path)

    tmp_contents = os.listdir(dst_chroma_path)
    if len(tmp_contents) == 0:
        print(f"Copying ChromaDB from {CHROMA_PATH} to {dst_chroma_path}")
        os.makedirs(dst_chroma_path, exist_ok=True)
        shutil.copytree(CHROMA_PATH, dst_chroma_path, dirs_exist_ok=True)
    else:
        print(f"✅ ChromaDB already exists in {dst_chroma_path}")

def get_runtime_chroma_path():
    if IS_USING_IMAGE_RUNTIME:
        return f"/tmp/{CHROMA_PATH}"
    else:
        return CHROMA_PATH