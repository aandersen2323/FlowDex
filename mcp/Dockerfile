FROM python:3.11-slim

WORKDIR /app

COPY mcp/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY mcp/mcp_http_server.py ./mcp_http_server.py

CMD ["python", "mcp_http_server.py"]
