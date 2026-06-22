import os # access env variables
import json
from dotenv import load_dotenv #load variable from .env file
from fastapi import FastAPI,HTTPException # error handling
from google import genai # Gemini SDK lets to talk to Gemini api
from pydantic import BaseModel,field_validator,ValidationError,StrictInt,StrictStr
from typing import Union
from google.genai.errors import ServerError

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

class UserProfile(BaseModel):
    name : StrictStr
    experience_years : StrictInt
    skills : list[StrictStr]

#Learning Function/Tool Calling
class TaskRequest(BaseModel):
    text : str

class CreateTaskArgs(BaseModel):
    title : str
    completed : bool
class ListTaskArgs(BaseModel):
    pass
class ToolCall(BaseModel):
    tool : StrictStr
    arguments : Union[CreateTaskArgs,ListTaskArgs]


tasks = []
#list of tools
def create_task(title:str,completed:bool):
    task = {
        "title" : title,
        "completed" : completed
    }
    tasks.append(task)
    return task
def list_tasks():
    return tasks

@app.post("/extract-profile", response_model=UserProfile)
def extract_profile(text:str):
    prompt = f"""
Extract the following information from the text.
Return ONLY valid JSON.
Do not incluce Markdown code fences.

Required Fields:
    name (string)
    experience_years (Integer)
    skills (array of strings)
Text: 
{text}
"""
    
    response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt)
    try: 
        data = json.loads(response.text) # convert json string to python object
        return UserProfile(**data)
    except json.JSONDecodeError: #ai doesnot return json string
        raise HTTPException(
            status_code=500,
            detail="AI returned invalid JSON"
        )
    except ValidationError: #raise when failed to pydantic model mismatch data types or miising keys
        raise HTTPException(
            status_code=500,
            detail="AI returned invalid data format"
        )

@app.post("/task-ai")
def task_ai(req:TaskRequest):
    prompt = f"""
    You are a task assistant.
    Choose the correct tool based on the user's request.
    
    Available Tools : 
        1. create_task
        Use when the user wants to create a new task.


        Schema : 
        {{
        "tool" : "create_task,
        "arguments":{{
            "title" : string,
            "completed" : boolean
        }}
        }}
        2. list_tasks
        Use when user wants to view all the tasks.
        Schema:
        {{
        "tool": "list_tasks",
        "arguments":{{}}
        }}

    Return only valid JSON.
    Donot include Markdown code fences
    Do not include Explanations

 

    user input: 
    {req.text}



    """
    
    # print(response.text)
    try:
        response = client.models.generate_content(model="gemini-2.5-flash",contents=prompt)
        data = json.loads(response.text)
        tool_call = ToolCall(**data)
        if tool_call.tool == "create_task":
            return create_task(
            title = tool_call.arguments.title,
            completed=tool_call.arguments.completed
        )
        elif tool_call.tool == "list_tasks":
            return list_tasks()
    except ServerError:
        raise HTTPException(
            status_code=503,
            detail="AI service is temporarily unavailable. Please try again later."
        )
    except json.JSONDecodeError: #ai doesnot return json string
        raise HTTPException(
            status_code=500,
            detail="AI returned invalid JSON"
        )
    except ValidationError: #raise when failed to pydantic model mismatch data types or miising keys
        raise HTTPException(
            status_code=500,
            detail="AI returned invalid data format"
        )


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
    
    