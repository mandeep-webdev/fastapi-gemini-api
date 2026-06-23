import os # access env variables
import json
from dotenv import load_dotenv #load variable from .env file
from fastapi import FastAPI,HTTPException # error handling
from google import genai # Gemini SDK lets to talk to Gemini api
from pydantic import BaseModel,field_validator,ValidationError,StrictInt,StrictStr
from typing import Union,Any
from google.genai.errors import ServerError
from google.genai import types
import math

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

#structured output
class Product(BaseModel):
    name : str
    price : int
    storeage : str
    color : str

class ProductRequest(BaseModel):
    text :str

@app.post("/structured-output")
def structured_output(req : ProductRequest):
    response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=req.text,
    # config gives extra instructions to gemini
    config=types.GenerateContentConfig(
        response_mime_type="application/json", #return json
        response_schema=Product
    )
)
    product = response.parsed # data is already parsed by gemini sdk and stored in parsed key
    return product
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
    


#Embedding

class EmbeddingReq(BaseModel):
    text : str

@app.post("/embedding-demo")
def embedding_demo(req:EmbeddingReq):
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=req.text
    )
    # Input --> {
    # "text": "React is a JavaScript library"
    # }
    # embeddings=[ContentEmbedding(
    # values=[
    #     -0.009860573,
    #     -0.007838764,
    #     0.002882608,
    #     -0.06735086,
    #     -0.02138334,
    #     <... 3067 more items ...>,
    # ]
    # )] 
    vector = response.embeddings[0].values
    print(len(vector))
    print(vector[:10])


# mini search-engine project like Google / ChatGPT retrieval system
documents = [
    "React is a JavaScript library for UI",
    "FastAPI is a Python backend framework",
    "Next.js supports SSR and routing",
    "Pizza is a popular Italian food"
]


def get_embedding(doc):
    return client.models.embed_content(
        model="gemini-embedding-001",
        contents=doc
    ).embeddings[0].values

# store embedding of each document
doc_vectors = []

#store vectors of docs 
for doc in documents:
    doc_vectors.append(get_embedding(doc))

#return similarity score range between 0 - 1
def cosine_similarity(a, b):
    dot = sum(x*y for x, y in zip(a, b))

    mag_a = math.sqrt(sum(x*x for x in a))
    mag_b = math.sqrt(sum(x*x for x in b))

    return dot / (mag_a * mag_b)

#retrive top k docs
def search(input):
    input_vec = get_embedding(input)
    scores = []
    for doc, vec in zip(documents,doc_vectors):
        score = cosine_similarity(input_vec,vec)
        scores.append((score,doc))
    scores.sort(reverse=True)
    return scores

results = search("frontend framework")
# pick top 2
top_docs = [doc for score, doc in results[:2]]

#give context to gemini
#Context = extra relevant text given to the LLM along with the question
context = "\n\n".join(top_docs) #double newline between docs

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=f"""
    Answer the question using the context below.
    
    Context: 
    {context}

    Question:
    frontend framework
"""
)
print(response.text)
# [(0.6332622974965787, 'React is a JavaScript library for UI'), 
#  (0.6328837774601989, 'FastAPI is a Python backend framework'), 
#  (0.5987360682634203, 'Next.js supports SSR and routing'), 
#  (0.5340642066240259, 'Pizza is a popular Italian food')
# ]

