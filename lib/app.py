from __future__ import annotations

from flask import Flask, jsonify, request

from lib.ai_client import generate_response
from lib.company_documents import COMPANY_DOCUMENTS
from lib.rag_service import build_prompt, retrieve_context, source_metadata


def create_app():
    app = Flask(__name__)

    @app.get("/api/health")
    def health_check():
        return jsonify({"status": "ok"})

    @app.post("/api/ask")
    def ask_question():
        """Accept a query and return a source-backed generated answer.

        TODO:
        1. Read JSON request data safely.
        2. Validate that `query` is a non-empty string.
        3. Retrieve relevant context from COMPANY_DOCUMENTS.
        4. If no context is found, return a safe fallback with an empty sources list.
        5. Build a structured prompt from the selected context.
        6. Call generate_response(prompt).
        7. Return query, answer, and sources as JSON.
        8. If generate_response raises RuntimeError, return a 503 service error.
        """
        #  1. Read JSON request data safely.
        payload = request.get_json(silent=True) or {}

        query = payload.get("query", "")

        include_prompt = bool(payload.get("include_prompt", False))

        #  2. Validate that `query` is a non-empty string.
        if not isinstance(query, str) or not query.strip():
            return (
                jsonify(
                    {
                        "error": "A non-empty 'query' string is required.",
                        "example": {"query": "How do I reset my password?"},
                    }
                ),
                400,
            )

        clean_query = query.strip()

        #  3. Retrieve relevant context from COMPANY_DOCUMENTS.
        context_matches = retrieve_context(
            query=query, documents=COMPANY_DOCUMENTS, limit=2
        )

        #  4. If no context is found, return a safe fallback with an empty sources list.
        if not context_matches:
            return (
                jsonify(
                    {
                        "query": clean_query,
                        "answer": (
                            "The approved support documents do not contain enough "
                            "information to answer that question."
                        ),
                        "sources": [],
                    }
                ),
                200,
            )

        #  5. Build a structured prompt from the selected context.
        sources = [source_metadata(match) for match in context_matches]

        prompt = build_prompt(clean_query, context_matches)

        #  6. Call generate_response(prompt).
        try:
            answer = generate_response(prompt)
        except RuntimeError as err:
            #  8. If generate_response raises RuntimeError, return a 503 service error.
            error_response = {
                "error": str(err),
                "message": (
                    "The API retrieved context and built a prompt, but the model "
                    "service did not return a usable response."
                ),
                "sources": sources,
            }
            if include_prompt:
                error_response["prompt"] = prompt

            return jsonify(error_response), 503

        #  7. Return query, answer, and sources as JSON.
        response_body = {"query": clean_query, "answer": answer, "sources": sources}

        if include_prompt:
            response_body["prompt"] = prompt

        return jsonify(response_body), 200

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
