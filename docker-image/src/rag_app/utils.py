from langchain_aws.embeddings.bedrock import BedrockEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import os
import sys
import shutil

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
        print("[DEBUG] get_chroma_db() starting")
        print(f"[DEBUG] CHROMA_PATH (env): {CHROMA_PATH}")
        print(f"[DEBUG] IS_USING_IMAGE_RUNTIME: {IS_USING_IMAGE_RUNTIME}")
        if IS_USING_IMAGE_RUNTIME:
            copy_chroma_to_tmp()

        # Debug source listing
        try:
            src_exists = os.path.exists(CHROMA_PATH)
            print(f"[DEBUG] source CHROMA_PATH exists: {src_exists} -> {CHROMA_PATH}")
            if src_exists:
                print("[DEBUG] source listing:", os.listdir(CHROMA_PATH)[:20])
        except Exception as _e:
            print(f"[DEBUG] error listing source CHROMA_PATH: {_e}")

        runtime_path = get_runtime_chroma_path()
        try:
            print(f"[DEBUG] runtime chroma path: {runtime_path}")
            print("[DEBUG] runtime listing:", os.listdir(runtime_path)[:20])
        except Exception as _e:
            print(f"[DEBUG] error listing runtime path: {_e}")

        # Prepare the DB.
        CHROMA_DB_INSTANCE = Chroma(
            persist_directory=runtime_path,
            embedding_function=get_embedding_function(),
        )

        print(f"Init ChromaDB {CHROMA_DB_INSTANCE} from {runtime_path}")
        try:
            items = CHROMA_DB_INSTANCE.get(include=[])
            ids = items.get("ids", [])
            print(f"[DEBUG] loaded ids count: {len(ids)}")
        except Exception as e:
            print(f"[DEBUG] error querying CHROMA DB on init: {e}")

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