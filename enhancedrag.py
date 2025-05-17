from sentence_transformers import SentenceTransformer, util
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from typing import List, Optional
from doc_loader import documents
import requests
import logging
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Futuro Chatbot RAG API", description="API for RAG (Retrieval-Augmented Generation) chatbot", version="1.0")

MODEL_NAME = "llama3:8b-instruct-q8_0"

prompt_instructions = """
Você é um atendente humano, prestativo e amigável. Seu objetivo é responder à consulta do usuário de forma natural e direta, como uma pessoa faria, baseando-se *exclusivamente* nas informações fornecidas nos textos.

Siga estas regras estritamente:

1.  **Persona:** Aja *apenas* como um atendente humano. **NUNCA** mencione que você é uma IA, um modelo de linguagem, um chatbot, ou que está processando informações/documentos/textos. Sua resposta deve soar como conhecimento próprio.
2.  **Base da Informação:** Use **APENAS E ESTRITAMENTE** as informações fornecidas nos textos recuperados para a *última* consulta do usuário. **NUNCA** invente, adicione dados externos, opiniões ou faça suposições não suportadas pelo texto.
3.  **Lidando com Ausência:** Se a resposta para a *última* consulta **não puder ser encontrada** nos textos fornecidos, diga educadamente que você não tem essa informação específica disponível. Use frases como: "Desculpe, não encontrei essa informação específica agora." ou "Não encontrei detalhes sobre esse ponto em particular."
4.  **Idioma:** Sua resposta deve ser **SEMPRE** em **Português do Brasil**. Se a *última* consulta do usuário for feita em outro idioma, responda naquele idioma, mantendo o tom amigável.
5.  **Foco e Contexto:** Use o histórico da conversa *apenas* para entender o contexto geral, mas responda **DIRETAMENTE** à *última* consulta do usuário, utilizando *apenas* as informações dos textos fornecidos *para essa consulta*. Não repita a pergunta do usuário.
6.  **Estilo:**
    * Mantenha um tom casual, amigável e prestativo.
    * Seja direto, claro e conciso. Evite respostas longas, divagações ou ambiguidades.
    * Seja confiante na informação fornecida (baseada no texto), evite frases como "Eu acho" ou "Eu acredito".
    * Use linguagem simples, evite jargões técnicos desnecessários.
    * Evite desculpas excessivas (além da instrução 3), avisos legais ou linguagem evasiva.
    * **NÃO FAÇA SUPOSIÇÕES** sobre o usuário ou o que ele/ela pode ser/fazer (ex: chamar de atleta).
7.  **Formato de Saída:**
    * Sua resposta deve ser **APENAS texto simples**.
    * **NUNCA** insira tags HTML, Markdown ou qualquer outra formatação de texto.
    * **NUNCA** insira caracteres especiais ou emojis.
"""

# Loading the model
try:
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise RuntimeError("Failed to load the model.")


# Embedding pre-processing
doc_embeddings = {}

try:
    logger.info(f"Calculating embeddings for {len(documents)} documents.")
    for doc in documents:
        doc_embeddings[doc["id"]] = model.encode(doc["text"], convert_to_tensor=True)
    logger.info("Document embeddings created successfully")
except Exception as e:
    logger.error(f"Error creating document embeddings: {e}")
    raise RuntimeError("Failed to create document embeddings.")


# Pydantic models for request and response
class ChatMessage(BaseModel):
    """Defines the structure of a chat message."""
    role: str = Field(..., description="Role of the message sender (user or assistant)")
    content: str = Field(..., description="Content of the message")

class ChatRequest(BaseModel):
    """Defines the structure of a chat request."""
    history: List[ChatMessage] = Field(..., description="Chat history, including user's last message.")
    max_docs: int = Field(2, ge=3, description="Maximum number of documents to retrieve for the query.")
    min_score: float = Field(0.4, ge=0.0, le=1.0, description="Minimum similarity score for document retrieval.")

class DocumentContext(BaseModel):
    """Defines the structure of a document context."""
    id: str
    score: float
    text: str

class ChatResponse(BaseModel):
    """Defines the structure of a chat response."""
    response: Optional[str] = Field(None, description="Response from the assistant.")
    used_context: List[DocumentContext] = Field([], description="List of documents used to generate the response.")
    error: Optional[str] = Field(None, description="Error message if any occurred during processing.")

# API endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat_rag(request: ChatRequest):
    """
    Processa uma mensagem de chat, recupera documentos relevantes,
    e gera uma resposta considerando o histórico e o contexto.
    """
    if not request.history:
        raise HTTPException(status_code=400, detail="O histórico da conversa não pode estar vazio.")

    last_user_message = next((msg for msg in reversed(request.history) if msg.role == 'user'), None)

    if not last_user_message:
        raise HTTPException(status_code=400, detail="Nenhuma mensagem de 'user' encontrada no histórico recente para usar como query.")

    current_query = last_user_message.content
    logger.info(f"Processando query: '{current_query}'")
    logger.info(f"Configuração: max_docs={request.max_docs}, min_score={request.min_score}")

    try:
        query_embedding = model.encode(current_query, convert_to_tensor=True)
    except Exception as e:
        logger.error(f"Erro ao gerar embedding para a query '{current_query}': {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao processar embedding da query: {e}")

    scored_docs = []
    for doc in documents:
        if doc["id"] not in doc_embeddings:
            logger.warning(f"Embedding não encontrado para doc_id: {doc['id']}. Pulando.")
            continue
        try:
            score_tensor = util.cos_sim(query_embedding, doc_embeddings[doc["id"]])
            score = score_tensor.item() # Extrai o valor float do tensor
            logger.debug(f"Doc ID: {doc['id']}, Score: {score:.4f}")
            if score >= request.min_score:
                scored_docs.append({"score": score, "doc": doc})
            else:
                logger.debug(f"Doc ID: {doc['id']} descartado (score {score:.4f} < {request.min_score})")
        except Exception as e:
            logger.error(f"Erro ao calcular similaridade para doc_id {doc['id']}: {e}")
            continue

    scored_docs.sort(key=lambda x: x["score"], reverse=True)

    selected_docs_data = scored_docs[:request.max_docs]
    selected_docs_context = [
        DocumentContext(id=item["doc"]["id"], score=item["score"], text=item["doc"]["text"])
        for item in selected_docs_data
    ]

    logger.info(f"Documentos selecionados ({len(selected_docs_context)}): {[d.id for d in selected_docs_context]} com scores {[f'{d.score:.4f}' for d in selected_docs_context]}")

    history_string = ""
    for message in request.history:
        role = getattr(message, 'role', 'unknown')
        content = getattr(message, 'content', '')
        history_string += f"{role.capitalize()}: {content}\n"
    history_string = history_string.strip()

    context_text = "\n\n---\n\n".join([doc.text for doc in selected_docs_context])
    if not context_text:
        context_text = "Nenhuma informação relevante encontrada nos documentos para esta consulta."
        logger.warning("Nenhum documento relevante encontrado ou selecionado.")

    prompt = f"""Você é um atendente humano prestativo e amigável auxiliando um cliente. Seu objetivo é responder à consulta do usuário de forma natural e direta, como uma pessoa faria, baseando-se *exclusivamente* nas informações fornecidas nos textos.

Use **APENAS** as informações fornecidas nos textos abaixo para formular sua resposta:
--- INÍCIO DAS INFORMAÇÕES ---
{context_text}
--- FIM DAS INFORMAÇÕES ---

Histórico da Conversa:
--- INÍCIO DO HISTÓRICO ---
{history_string}
--- FIM DO HISTÓRICO ---

Regras para sua resposta:
--- REGRAS PARA RESPOSTA ---
{prompt_instructions}
--- FIM DAS REGRAS ---

Consulta Atual do Usuário:
--- INÍCIO DA CONSULTA ---
{current_query}
--- FIM DA CONSULTA ---

Sua Resposta (como atendente):
"""

    logger.debug(f"Prompt enviado para o LLM:\n{prompt}")

    try:
        llm_url = "http://localhost:11434/api/generate"
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            # "options": {
            #    "temperature": 0.7
            # }
        }
        response = requests.post(llm_url, json=payload, timeout=120)
        response.raise_for_status()

        response_data = response.json()
        llm_response_content = response_data.get("response")

        if not llm_response_content:
            logger.warning("LLM retornou uma resposta vazia.")
            llm_response_content = "Não consegui gerar uma resposta no momento."

        logger.info(f"Resposta recebida do LLM: '{llm_response_content[:100]}...'")

        return ChatResponse(
            response=llm_response_content.strip(),
            used_context=selected_docs_context
        )

    except requests.exceptions.RequestException as e:
        logger.error(f"Erro na comunicação com o LLM em {llm_url}: {e}")
        raise HTTPException(status_code=502, detail=f"Erro ao contatar o serviço de LLM: {e}")
    except Exception as e:
        logger.error(f"Erro inesperado ao processar a resposta do LLM: {e}")
        return ChatResponse(
            error=f"Erro ao processar a resposta do LLM: {e}",
            used_context=selected_docs_context
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)