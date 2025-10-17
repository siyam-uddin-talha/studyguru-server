import os
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import uvicorn

# Load environment variables from .env file
load_dotenv()

# Ensure the Google API key is set
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY environment variable not set")

# 1. Initialize the FastAPI app
app = FastAPI(
    title="Gemini 2.5 Pro Streaming API",
    description="An API for streaming responses from Gemini 2.5 Pro using FastAPI and LangChain.",
)

# 2. Set up the LangChain model
# We use .astream() later, which automatically enables streaming
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

# Optional: Create a prompt template for a more structured input
prompt = ChatPromptTemplate.from_messages([("human", "{question}")])

# Combine the prompt and the model into a runnable chain
chain = prompt | llm


# 3. Define the async generator for streaming
async def stream_llm_response(question: str):
    """
    This async generator function streams the response from the LLM.
    It uses LangChain's .astream() method to get tokens as they are generated.
    """
    try:
        # astream() returns an async iterator of response chunks
        async for chunk in chain.astream({"question": question}):
            # Each chunk has a 'content' attribute with the text
            if chunk.content:
                yield chunk.content
    except Exception as e:
        print(f"An error occurred during streaming: {e}")
        yield "Error: Could not process the request."


# 4. Define the FastAPI endpoint
@app.post("/stream-gemini")
async def stream_gemini(question: str):
    """
    Endpoint to receive a question and stream back the Gemini model's response.
    """
    # Return a StreamingResponse, which takes an async generator and a media type
    return StreamingResponse(stream_llm_response(question), media_type="text/plain")


# To run the app (optional, can be done from the command line)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
