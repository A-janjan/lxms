from langchain_google_genai import ChatGoogleGenerativeAI

GEMINI_MODEL_NAME = "gemini-1.5-flash"

gemini_chat = ChatGoogleGenerativeAI(model=GEMINI_MODEL_NAME)