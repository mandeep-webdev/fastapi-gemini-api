import os # access env variables
from dotenv import load_dotenv #load variable from .env file
from fastapi import FastAPI,HTTPException # error handling
from google import genai # Gemini SDK lets to talk to Gemini api
from pydantic import BaseModel,field_validator

load_dotenv() #read .env file

app = FastAPI()
# with this we connect to gemini
client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
    )

class Request(BaseModel):
    question : str

    @field_validator("question") #custom validator
    @classmethod
    def validate_question(cls,value:str):
        if not value.strip(): #remove extra spaces from front and back
            raise ValueError("Question cannot be empty")
        return value


class Response(BaseModel):
    answer : str

@app.post("/ask", response_model=Response)
def ask_ai(req:Request):
    system_prompt = """
You are an Ai engineering mentor for beginner developers.
You specialize in python backend development with FASTAPI and Ai-engineering fundamentals.


Explain concepts simply.
Use practical examples.
keep response concise.

"""
    prompt = f"""
{system_prompt}
User question : {req.question}




"""
    try:
        # if connection failed or authentication failed python go to except block
        res = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
        )
        # if response is empty
        if not res.text:
            raise HTTPException(
                status_code=502,
                detail="AI service returned an empty response"
            )
        return Response(answer=res.text)
        

    except Exception:
        raise HTTPException(
            status_code=503,
            detail="AI service unavailable"
        )
    
    