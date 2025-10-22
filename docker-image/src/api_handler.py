import uvicorn
import re
import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from mangum import Mangum
from pydantic import BaseModel, Field, field_validator
from rag_app.query import query_rag, QueryResponse

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


app = FastAPI()

# Configure CORS - use environment variable for allowed origins
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").split(",") if os.getenv("ALLOWED_ORIGINS") else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False, 
    allow_methods=["GET", "POST"], 
    allow_headers=["Content-Type"],
)

handler = Mangum(app)  # Entry point for AWS Lambda.


class SubmitQueryRequest(BaseModel):
    """Request model for submitting a query to the RAG system.

    Validates that queries are:
    - Non-empty
    - Within reasonable length limits
    - Contain safe characters only
    """
    query_text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The question to ask the RAG system",
        examples=["What is the maximum speed?", "How can I move the seat forwards?"]
    )

    @field_validator('query_text')
    @classmethod
    def validate_query_text(cls, v: str) -> str:
        """Validate and sanitize query text.

        Args:
            v: The query text to validate

        Returns:
            Cleaned and validated query text

        Raises:
            ValueError: If query contains invalid characters or patterns
        """
        v = v.strip()

        # Check if empty after stripping
        if not v:
            raise ValueError("Query cannot be empty or only whitespace")

        # Very basic defense against prompt injection, trying to stop things like using {system} tags
        allowed_pattern = r'^[a-zA-Z0-9\s\?.!,;:\'\"\-\(\)\/]+$'
        if not re.match(allowed_pattern, v):
            raise ValueError(
                "Query contains invalid characters. "
                "Only letters, numbers, spaces, and basic punctuation are allowed."
            )

        return v


@app.get("/", response_class=HTMLResponse)
def index():
    """Serve the Gradio-style interface at the root path"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Query Interface</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            max-width: 1200px;
            width: 100%;
            padding: 40px;
        }

        h1 {
            color: #333;
            margin-bottom: 10px;
            font-size: 2.5em;
        }

        .subtitle {
            color: #666;
            margin-bottom: 30px;
            font-size: 1.1em;
        }

        .query-section {
            margin-bottom: 30px;
        }

        label {
            display: block;
            color: #555;
            font-weight: 600;
            margin-bottom: 8px;
        }

        textarea {
            width: 100%;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 16px;
            font-family: inherit;
            resize: vertical;
            transition: border-color 0.3s;
        }

        textarea:focus {
            outline: none;
            border-color: #667eea;
        }

        .button-group {
            display: flex;
            gap: 10px;
            margin-bottom: 30px;
        }

        button {
            flex: 1;
            padding: 15px 30px;
            font-size: 16px;
            font-weight: 600;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.3s;
        }

        .submit-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }

        .submit-btn:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
        }

        .submit-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }

        .clear-btn {
            background: #e0e0e0;
            color: #555;
        }

        .clear-btn:hover {
            background: #d0d0d0;
        }

        .results {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
        }

        .result-box {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            min-height: 200px;
        }

        .result-box h3 {
            color: #555;
            margin-bottom: 15px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .copy-btn {
            background: #667eea;
            color: white;
            border: none;
            padding: 5px 12px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 12px;
        }

        .copy-btn:hover {
            background: #5568d3;
        }

        .result-content {
            color: #333;
            line-height: 1.6;
            white-space: pre-wrap;
            word-wrap: break-word;
        }

        .loading {
            display: none;
            text-align: center;
            color: #667eea;
            font-weight: 600;
            margin: 20px 0;
        }

        .loading.active {
            display: block;
        }

        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .error {
            background: #fee;
            color: #c33;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            display: none;
        }

        .error.active {
            display: block;
        }

        @media (max-width: 768px) {
            .results {
                grid-template-columns: 1fr;
            }

            h1 {
                font-size: 2em;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>RAG Question Answering System</h1>
        <p class="subtitle">Ask questions about your documents and get answers based on the knowledge base.</p>

        <div class="error" id="error"></div>

        <div class="query-section">
            <label for="queryInput">Your Question:</label>
            <textarea id="queryInput" rows="4" placeholder="Enter your question here..."></textarea>
        </div>

        <div class="button-group">
            <button class="submit-btn" id="submitBtn" onclick="submitQuery()">Submit Query</button>
            <button class="clear-btn" onclick="clearAll()">Clear</button>
        </div>

        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Processing your query...</p>
        </div>

        <div class="results" id="results" style="display: none;">
            <div class="result-box">
                <h3>
                    Answer
                    <button class="copy-btn" onclick="copyToClipboard('answer')">Copy</button>
                </h3>
                <div class="result-content" id="answer"></div>
            </div>
            <div class="result-box">
                <h3>
                    Sources
                    <button class="copy-btn" onclick="copyToClipboard('sources')">Copy</button>
                </h3>
                <div class="result-content" id="sources"></div>
            </div>
        </div>
    </div>

    <script>
        // Auto-detect API URL (same domain as the page)
        const API_URL = window.location.origin + '/submit_query';

        // Allow Enter key in textarea with Shift+Enter for new line
        document.getElementById('queryInput').addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                submitQuery();
            }
        });

        function setQuery(text) {
            document.getElementById('queryInput').value = text;
        }

        function clearAll() {
            document.getElementById('queryInput').value = '';
            document.getElementById('results').style.display = 'none';
            document.getElementById('error').classList.remove('active');
        }

        async function submitQuery() {
            const queryText = document.getElementById('queryInput').value.trim();
            const errorDiv = document.getElementById('error');
            const loadingDiv = document.getElementById('loading');
            const resultsDiv = document.getElementById('results');
            const submitBtn = document.getElementById('submitBtn');

            // Validation
            if (!queryText) {
                showError('Please enter a question');
                return;
            }

            // Clear previous results
            errorDiv.classList.remove('active');
            resultsDiv.style.display = 'none';

            // Show loading
            loadingDiv.classList.add('active');
            submitBtn.disabled = true;

            try {
                const response = await fetch(API_URL, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        query_text: queryText
                    })
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const data = await response.json();

                // Display results
                document.getElementById('answer').textContent = data.response_text || 'No answer received';

                const sources = data.sources || [];
                document.getElementById('sources').textContent = sources.length > 0
                    ? sources.map(s => `• ${s}`).join('\\n')
                    : 'No sources found';

                resultsDiv.style.display = 'grid';

            } catch (error) {
                console.error('Error:', error);
                showError(`Failed to query the API: ${error.message}`);
            } finally {
                loadingDiv.classList.remove('active');
                submitBtn.disabled = false;
            }
        }

        function showError(message) {
            const errorDiv = document.getElementById('error');
            errorDiv.textContent = message;
            errorDiv.classList.add('active');
        }

        function copyToClipboard(elementId) {
            const text = document.getElementById(elementId).textContent;
            navigator.clipboard.writeText(text).then(() => {
                // Visual feedback
                const btn = event.target;
                const originalText = btn.textContent;
                btn.textContent = 'Copied!';
                setTimeout(() => {
                    btn.textContent = originalText;
                }, 2000);
            });
        }
    </script>
</body>
</html>
"""


@app.post("/submit_query", response_model=QueryResponse)
async def submit_query_endpoint(request: SubmitQueryRequest) -> QueryResponse:
    """Submit a query to the RAG system.

    Args:
        request: The query request containing the query text

    Returns:
        QueryResponse with the answer and source citations

    Raises:
        HTTPException: 400 for validation errors, 500 for server errors
    """
    try:
        query_response = query_rag(request.query_text)
        return query_response

    except ValueError as e:
        # Invalid input (though this should be caught by Pydantic)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid query: {str(e)}"
        )

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred processing your query. Please try again."
        )

if __name__ == "__main__":
    # Run this as a server directly.
    port = 8000
    logger.info(f"Starting FastAPI server on port {port}")
    uvicorn.run("api_handler:app", host="0.0.0.0", port=port)