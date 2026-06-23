import os # access env variables
import json
from dotenv import load_dotenv #load variable from .env file
from fastapi import FastAPI,HTTPException # error handling
from google import genai # Gemini SDK lets to talk to Gemini api
from pydantic import BaseModel,field_validator,ValidationError,StrictInt,StrictStr
from typing import Union,Any
from google.genai.errors import ServerError
from google.genai import types

load_dotenv() #read .env file

app = FastAPI()
# with this we connect to gemini
client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
    )


#Learning Function/Tool Calling
class TaskRequest(BaseModel):
    text : str


tasks = []
task_id_counter = 1

# create tool declaration
# i am describing my tools here
create_task_function = types.FunctionDeclaration(
    name= "create_task",
    description="Create a new task", #This helps Gemini decide when to use the function.
    parameters_json_schema={ #This defines the function inputs.
        "type" : "object", #The function arguments are a dictionary.
        "properties" : { #This lists all available arguments.
            "title" : {
                "type" : "string",
                "description" : "title of the task"
            },
            "completed" : {
                "type" : "boolean",
                "description" : "Whether the task is completed"
            }
        },
        "required" : ["title","completed"] #This means Gemini must provide both fields.
    })
list_tasks_function = types.FunctionDeclaration(
    name="list_tasks",
    description= "shows or lists all tasks",
    parameters_json_schema= {
        "type" : "object",
        "properties" : {
        }
    }
)
delete_task_function = types.FunctionDeclaration(
    name="delete_task",
    description="delete a task",
    parameters_json_schema= {
        "type" : "object",
        "properties" : {
            "task_id" : {
                "type" : "integer",
                "description" : "id of the task to delete"
            }
        },
        "required" : ["task_id"]
    }
    
)
update_task_function = types.FunctionDeclaration(
    name="update_task",
    description= "update a task",
    parameters_json_schema= {
        "type" : "object",
        "properties" : {
            "task_id" : {
                "type" : "integer",
                "description" : "id of the task to be updated"
            },
            "title": {
                "type" : "string",
                "description" : "title of the task"
            },
            "completed" : {
                "type" : "boolean",
                "description" : "whether the task is completed"
            }
        },
        "required" : ["task_id","title","completed"]
    }
)
task_tool = types.Tool(
    function_declarations=[create_task_function,list_tasks_function,delete_task_function,update_task_function]
)


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

TOOLS = {
    "create_task" : create_task,
    "list_tasks" : list_tasks,
    "delete_task" : delete_task,
    "update_task" : update_task
}
@app.post("/task-native")
def task_native(req: TaskRequest):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=req.text,
        config=types.GenerateContentConfig(
        tools=[task_tool]
    ))
    # gemini return function_call obj
    # Gemini returns function_call=FunctionCall(
        #   args={
        #     'completed': False,
        #     'title': 'learn FastAPI'
        #   },
        #   name='create_task'
        # ),
    function_call = response.function_calls[0]
    fn = TOOLS[function_call.name]
    return fn(**function_call.args)
    

   