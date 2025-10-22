import logging
from langchain_aws import ChatBedrock
from query import query_rag

# Configure logging
logger = logging.getLogger(__name__)

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? 
"""


def test_query_for_numbers():
    assert query_and_validate(
        question="What is the measuring range of the outside temperature display?",
        expected_response="The measuring range lies between -40℃ (-40°F) and +50 ℃ (+122°F).",
    )

def query_and_validate(question: str, expected_response: str):
    response_text = query_rag(question)
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text
    )

    model = ChatBedrock(model_id="amazon.titan-text-lite-v1")
    evaluation_results_str = model.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

    logger.debug(prompt)

    if "true" in evaluation_results_str_cleaned:
        # Log response if it is correct.
        logger.info(f"✓ Response validated: {evaluation_results_str_cleaned}")
        return True
    elif "false" in evaluation_results_str_cleaned:
        # Log response if it is incorrect.
        logger.warning(f"✗ Response validation failed: {evaluation_results_str_cleaned}")
        return False
    else:
        raise ValueError(
            f"Invalid evaluation result. Cannot determine if 'true' or 'false'."
        )