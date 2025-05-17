from fastapi import FastAPI, HTTPException
from typing import Dict, Any
import uvicorn
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

rag_url = "http://localhost:8001/chat"

@app.get("/")
def index():
    """1
    Simple index route to check if the server is running.
    """
    return {"message": "Webhook Server is running!"}

@app.post("/webhook")
async def webhook(payload: Dict[str, Any]):
    """
    Webhook endpoint to receive and process incoming requests.
    """
    logger.info("Received webhook request.")
    
    if not payload:
        logger.warning("Empty payload received.")
        raise HTTPException(status_code=400, detail="Empty payload received.")
    
    entry_list = payload.get("entry")
    
    if not isinstance(entry_list, list) or not entry_list:
        logger.warning("Payload 'entry' is not a list or is empty, or is missing.")
        raise HTTPException(status_code=400, detail="Invalid entry list received.")
    
    entry_data = entry_list[0]
    
    changes_list = entry_data.get("changes")
    
    if not isinstance(changes_list, list) or not changes_list:
        logger.warning("Payload 'changes' is not a list or is empty, or is missing.")
        raise HTTPException(status_code=400, detail="Invalid changes list received.")
    
    change_data = changes_list[0]
    
    value_data = change_data.get("value")
    
    if not isinstance(value_data, dict):
        logger.warning("Payload 'value' is not a dictionary or is missing.")
        raise HTTPException(status_code=400, detail="Invalid value data received.")
    
    messages_list = value_data.get("messages")
    
    if not isinstance(messages_list, list) or not messages_list:
        logger.warning("Payload 'messages' is not a list or is empty, or is missing.")
        raise HTTPException(status_code=400, detail="Invalid messages list received.")
    
    last_message = messages_list[-1]
    
    message_type = last_message.get("type")
    sender_phone = last_message.get("from")
    message_id = last_message.get("id")
    timestamp = last_message.get("timestamp")
    message_body = None
    
    if message_type == "text":
        text_data = last_message.get("text")
        if isinstance(text_data, dict):
            message_body = text_data.get("body")
        else:
            logger.warning("Payload 'text' is not a dictionary or is missing.")
            raise HTTPException(status_code=400, detail="Invalid text data received.")
    
    if message_body is None:
        logger.warning(f"Could not extract text body from message ID : {message_id}. Type was: {message_type}")
        return {"status": "ignored", "message": f"No processable text message body found for message ID: {message_id}"}, 200
    
    rag_history = [
        {
            "role": "user",
            "content": message_body
        }
    ]
    
    rag_payload = {
        "history": rag_history,
        "max_docs": 3,
        "min_score": 0.5
    }
    
    logger.info(f"Sending payload to RAG: {rag_payload}")
    
    rag_response = send_to_rag(rag_url, rag_payload)
    
    if rag_response is None:
        logger.error("Failed to get a response from RAG.")
        raise HTTPException(status_code=500, detail={"error": "Failed to process message via RAG service"})
    
    logger.info(f"Received response from RAG: {rag_response}")
    return {"status": "success", "message": "Message processed via RAG", "rag_response": rag_response}, 200

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

def send_to_rag(url: str, payload: Dict[str, Any]):
    """
    Function to send the payload to the RAG service.
    """
    try:
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error sending to RAG: {e}")
        return None