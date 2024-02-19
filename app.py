from flask import Flask, request, jsonify
from service import QueryService, DocumentProcessingService
from user import User
from pathlib import Path
from utils import load_config

config = load_config()

app = Flask(__name__)

document_processing_service = DocumentProcessingService()
user_class = User()


@app.route("/", methods=["GET", "POST"])
def home():
    """Endpoint to greet the users."""
    return jsonify({"Greet": "Welcome to Bot 0.01", "message": "Success."}), 200


@app.route("/upload", methods=["POST"])
def handle_upload():
    user_id = request.form.get("user_id", None)
    conversation_id = request.form.get("conversation_id", None)
    if user_id is None:
        user_id, conversation_id = user_class.create_new_user()

    file = request.files.get("uploaded_files")
    if file:
        document_processing_service.process_uploaded_file(
            file, user_id, conversation_id
        )
        return (
            jsonify(
                {
                    "message": "File uploaded successfully.",
                    "user_id": user_id,
                    "conversation_id": conversation_id,
                }
            ),
            200,
        )
    else:
        return jsonify({"error": "No file uploaded."}), 400


@app.route("/response", methods=["POST"])
def handle_query():
    data = request.json
    user_id = data.get("user_id")
    conversation_id = data.get("conversation_id")
    query_text = data.get("query_text")

    if user_id and conversation_id and query_text:
        if user_id not in document_processing_service.user_conversation_memories:
            document_processing_service.initialize_conversation_memory(
                user_id, conversation_id
            )

        document_processing_service.get_conversation_memory(user_id, conversation_id)
        conversation_memory = document_processing_service.get_conversation_memory(
            user_id, conversation_id
        )
        db_path = Path(config["vector_store_dir"], user_id, conversation_id)

        query_service = QueryService(db_path=db_path)
        output, reranked_docs = query_service.process_query(
            query_text, conversation_memory
        )

        response = {
            "message": "Success.",
            "user_id": user_id,
            "conversation_id": conversation_id,
            "query_text": query_text,
            "reranked_documents": [
                {
                    f"documents_{idx}": doc.page_content,
                    "metadata": {
                        **doc.metadata,
                        "score": float(doc.metadata.get("score", 0)),
                    },
                }
                for idx, doc in enumerate(reranked_docs)
            ],
            "response": {
                "human_input": output.get("human_input", None),
                "chat_history": output.get("chat_history", None),
                "output_text": output.get("output_text", None),
            },
        }

        return jsonify(response), 200
    else:
        return jsonify({"error": "Session ID and query text are required."}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)
