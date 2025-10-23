from langchain_aws.embeddings.bedrock import BedrockEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import os
import sys
import shutil
import logging

# Configure logging
logger = logging.getLogger(__name__)

CHROMA_DB_INSTANCE = None  # Reference to singleton instance of ChromaDB
CHROMA_PATH = os.environ.get("CHROMA_PATH", "src/data/chroma")
IS_USING_IMAGE_RUNTIME = bool(os.environ.get("IS_USING_IMAGE_RUNTIME", False))

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
        logger.info("Successfully connected to AWS Bedrock embeddings")
        return embeddings
    except Exception as e:
        logger.error(f"AWS Bedrock embeddings error: {str(e)}")
        raise e


def get_chroma_db():
    global CHROMA_DB_INSTANCE
    if not CHROMA_DB_INSTANCE:
        logger.debug("get_chroma_db() starting")
        logger.debug(f"CHROMA_PATH (env): {CHROMA_PATH}")
        logger.debug(f"IS_USING_IMAGE_RUNTIME: {IS_USING_IMAGE_RUNTIME}")
        if IS_USING_IMAGE_RUNTIME:
            copy_chroma_to_tmp()

        runtime_path = get_runtime_chroma_path()

        # Prepare the DB.
        CHROMA_DB_INSTANCE = Chroma(
            persist_directory=runtime_path,
            embedding_function=get_embedding_function(),
        )

        logger.info(f"Initialized ChromaDB from {runtime_path}")
        try:
            items = CHROMA_DB_INSTANCE.get(include=[])
            ids = items.get("ids", [])
            logger.info(f"Loaded {len(ids)} documents from ChromaDB")
        except Exception as e:
            logger.error(f"Error querying ChromaDB on init: {e}")

    return CHROMA_DB_INSTANCE

def copy_chroma_to_tmp():
    dst_chroma_path = get_runtime_chroma_path()

    if not os.path.exists(dst_chroma_path):
        os.makedirs(dst_chroma_path)

    tmp_contents = os.listdir(dst_chroma_path)
    if len(tmp_contents) == 0:
        logger.info(f"Copying ChromaDB from {CHROMA_PATH} to {dst_chroma_path}")
        os.makedirs(dst_chroma_path, exist_ok=True)
        shutil.copytree(CHROMA_PATH, dst_chroma_path, dirs_exist_ok=True)
    else:
        logger.info(f"ChromaDB already exists in {dst_chroma_path}")

def get_runtime_chroma_path():
    if IS_USING_IMAGE_RUNTIME:
        return f"/tmp/{CHROMA_PATH}"
    else:
        return CHROMA_PATH