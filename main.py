import os # access env variables
import json
from dotenv import load_dotenv #load variable from .env file
from fastapi import FastAPI,HTTPException # error handling
from google import genai # Gemini SDK lets to talk to Gemini api
from pydantic import BaseModel,field_validator,ValidationError,StrictInt,StrictStr
from typing import Union,Any
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
class DeleteTaskArgs(BaseModel):
    task_id : int
class UpdateTaskArgs(BaseModel):
    task_id: int
    title: str
    completed: bool
class ToolCall(BaseModel):
    tool : StrictStr
    arguments : dict[str,Any]


tasks = []
task_id_counter = 1
#list of tools 
def create_task(title:str,completed:bool):
    global task_id_counter
    task = {
        "id" : task_id_counter,
        "title" : title,
        "completed" : completed
    }
    tasks.append(task)
    task_id_counter+=1
    return task
def list_tasks():
    return tasks

def delete_task(task_id:int):
    global tasks
    tasks = [task for task in tasks if task["id"] != task_id]
    return {"message": "Task deleted"}

def update_task(task_id: int, title: str, completed: bool):
    for task in tasks:
        if task["id"] == task_id:
            task["title"] = title
            task["completed"] = completed

            return task

    return {"error": "Task not found"}
SCHEMAS = {
    "create_task" : CreateTaskArgs,
    "list_tasks" : ListTaskArgs,
    "delete_task" : DeleteTaskArgs,
    "update_task" : UpdateTaskArgs
}
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
        3. delete_task
        Use when user wants to delete or remove a task
        Schema : {{
        "tool": "delete_task",
        "arguments": {{
         "task_id" : integer
        }}
        }}
        4. update_task
        Use when user wants to modify an existing task
        Schema : {{
        "tool" : "update_task",
        "arguments" : {{
        "task_id" : integer,
        "title" : string,
        "completed" : "boolean"
        }}
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
        print(data)
        # data = {'tool': 'create_task', 
        # 'arguments': {'title': 'learn python', 'completed': False}}
        tool_call = ToolCall(**data) 
        # tool_call = {"tool" = "create_task", "argumanets" = {}}
        validation_args_schema = SCHEMAS.get(tool_call.tool) #return schema class
        validated_args = validation_args_schema(**tool_call.arguments)
        TOOLS = {
            "create_task" : create_task,
            "list_tasks" : list_tasks,
            "delete_task" : delete_task,
            "update_task" : update_task
        }
        tool_fn = TOOLS.get(tool_call.tool)
        #Pydantic method model_dump() that converts  Pydantic model into normal Python dictionary.
        return tool_fn(**validated_args.model_dump())
        
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
    
    