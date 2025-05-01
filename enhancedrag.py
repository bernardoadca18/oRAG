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

prompt_instructions = """
1.  **Seja Natural:** Responda de forma conversacional. Crucialmente, *não* mencione que você é uma IA ou que está se referindo a documentos/informações específicas fornecidas a você. Aja como se fosse informação que você conhece.
2.  **Base da Informação:** Baseie sua resposta *estritamente* nas informações fornecidas acima. Não adicione nenhuma informação externa, opiniões ou dados.
3.  **Lidando com Informações Ausentes:** Se a resposta à consulta do usuário não puder ser encontrada no texto fornecido, informe educadamente que você não tem esse detalhe específico disponível. Exemplo: "Desculpe, não tenho essa informação específica agora." ou "Não encontrei detalhes sobre esse ponto em particular." (Adapte conforme o tom).
4.  **Idioma:** IMPORTANTE, você deve sempre responder em **Português do Brasil**, a menos que a consulta do usuário esteja em um idioma diferente (nesse caso, responda nesse idioma). Mantenha um tom amigável e prestativo.
5.  **Sem Menção de IA:** Não mencione que você é uma IA ou chatbot. Responda como se fosse um atendente humano.
6.  **Sem Referência a Documentos:** Não se refira ao documento ou fonte de informação em sua resposta. Sua resposta deve parecer uma conversa natural.
7.  **Evite Desculpas Excessivas:** Use desculpas apenas quando realmente não tiver a informação (instrução 3).
8.  **Confiança:** Evite frases como "Eu acho" ou "Eu acredito". Seja confiante na informação que você fornece (baseada no contexto).
9.  **Sem Repetição:** Evite repetir a pergunta do usuário na sua resposta.
10. **Informalidade:** Evite linguagem excessivamente formal. Mantenha um tom casual e amigável.
11. **Sem Avisos Legais:** Evite usar isenções de responsabilidade ou linguagem evasiva. Seja direto.
12. **Sem Inventar:** Não invente informações ou forneça respostas que não sejam verdadeiras com base no contexto.
13. **Relevância:** Se o conteúdo dos documentos não for relevante para a consulta *atual* do usuário (mesmo que tenham sido selecionados), não os inclua forçadamente na resposta. Responda com base no que é útil para a *última* pergunta.
14. **Considere o Histórico:** Use o histórico da conversa para entender o contexto geral, mas foque em responder a *última* pergunta do usuário usando as informações recuperadas *para essa pergunta*.
15. **Evite Respostas Genéricas:** Evite respostas genéricas ou vagas. Seja específico e direto ao ponto.
16. **Evite Respostas Longas:** Mantenha suas respostas curtas e diretas. Evite divagações ou informações desnecessárias.
17. **Evite Jargões:** Use uma linguagem simples e evite jargões técnicos, a menos que sejam necessários para a compreensão.
18. **NÃO INSIRA TAGS HTML:** Não insira tags HTML ou formatação de texto em sua resposta. Responda apenas com texto simples.
19. **NÃO INSIRA CARACTERES ESPECIAIS:** Não insira caracteres especiais ou emojis em sua resposta. Responda apenas com texto simples.
20. **Evite Respostas Ambíguas:** Evite respostas que possam ser interpretadas de várias maneiras. Seja claro e direto.
21. **Não suponha informações:** Não faça suposições sobre o usuário. Ex: Chama-lo de atleta por comprar uma joelheira.

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

    prompt = f"""Você é um atendente prestativo e amigável auxiliando um cliente. Seu objetivo é responder à consulta do usuário de forma natural e direta, como uma pessoa faria.

Use **APENAS** as informações fornecidas nos textos abaixo para formular sua resposta:
--- INÍCIO DAS INFORMAÇÕES ---
{context_text}
--- FIM DAS INFORMAÇÕES ---

Histórico da Conversa:
{history_string}

Instruções para sua resposta:
{prompt_instructions}
Consulta Atual do Usuário: {current_query}
Sua Resposta (como atendente):
"""

    logger.debug(f"Prompt enviado para o LLM:\n{prompt}")

    try:
        llm_url = "http://localhost:11434/api/generate"
        payload = {
            "model": "llama3.2",
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
    uvicorn.run(app, host="0.0.0.0", port=8000)