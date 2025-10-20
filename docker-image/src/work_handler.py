import json
from typing import Any, Dict, List
from rag_app.query_response_model import QueryResponseModel
from rag_app.query import query_rag
import time

def handler(event, context):
    """
    SQS-triggered worker.
    If event has 'Records' -> SQS batch; else treat it as a single job dict.
    """
    # SQS batch
    if "Records" in event:
        results: List[Dict[str, Any]] = []
        for rec in event["Records"]:
            job = json.loads(rec["body"])
            results.append(_process_job(job))
        return {"batch_count": len(results), "results": results}
    # Direct/local
    return _process_job(event if isinstance(event, dict) else json.loads(event))

def _process_job(job: Dict[str, Any]):
    query_id = job["query_id"]
    query_text = job["query_text"]
    try:
        rag = query_rag(query_text)
        # Mark SUCCEEDED on the record using your model
        q = QueryResponseModel.get_item(query_id)
        q.response_text = rag.response_text
        q.sources = rag.sources
        q.status = "SUCCEEDED"
        q.put_item()
        return {"query_id": query_id, "status": "SUCCEEDED"}
    except Exception as e:
        try:
            q = QueryResponseModel.get_item(query_id)
            q.status = "FAILED"
            q.put_item()
        except Exception:
            pass
        return {"query_id": query_id, "status": "FAILED", "error": str(e)}