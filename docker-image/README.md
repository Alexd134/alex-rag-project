Start by uploading pdfs into the chroma db - this is done by:
- adding files into the docker-image/data/source folder
- cd into docker-images 
- run 'uv run add_to_database'

Query locally using 'uv run query {query_text}'

or to deploy to aws lambda:
- 'docker build -t aws_rag_app .'
- 'cdk deploy'  
- query from the /docs fast api interface


TODO add conversational memory
add a simple gradio web interface to allow adding of items 
and submission of queries
allow different doc types to be used