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
import ollama
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings,ChatOllama
from langchain_core.prompts import PromptTemplate
import faiss
import numpy as np

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
    # print(len(vector))
    # print(vector[:10])


# mini search-engine project like Google / ChatGPT retrieval system
documents = [
    "React is a JavaScript library for UI",
    "FastAPI is a Python backend framework",
    "Next.js supports SSR and routing",
    "Pizza is a popular Italian food"
]


# def get_embedding(doc):
#     return client.models.embed_content(
#         model="gemini-embedding-001",
#         contents=doc
#     ).embeddings[0].values

# store embedding of each document
doc_vectors = []

#store vectors of docs 
# for doc in documents:
#     doc_vectors.append(get_embedding(doc))

#return similarity score range between 0 - 1
def cosine_similarity(a, b):
    dot = sum(x*y for x, y in zip(a, b))

    mag_a = math.sqrt(sum(x*x for x in a))
    mag_b = math.sqrt(sum(x*x for x in b))

    return dot / (mag_a * mag_b)

#retrive top k docs
# def search(input):
#     input_vec = get_embedding(input)
#     scores = []
#     for doc, vec in zip(documents,doc_vectors):
#         score = cosine_similarity(input_vec,vec)
#         scores.append((score,doc))
#     scores.sort(reverse=True)
#     return scores

# results = search("frontend framework")
# pick top 2
# top_docs = [doc for score, doc in results[:2]]

#give context to gemini
#Context = extra relevant text given to the LLM along with the question
# context = "\n\n".join(top_docs) #double newline between docs

# response = client.models.generate_content(
#     model="gemini-2.5-flash",
#     contents=f"""
#     Answer the question using the context below.
    
#     Context: 
#     {context}

#     Question:
#     frontend framework
# """
# )
# print(response.text)
# [(0.6332622974965787, 'React is a JavaScript library for UI'), 
#  (0.6328837774601989, 'FastAPI is a Python backend framework'), 
#  (0.5987360682634203, 'Next.js supports SSR and routing'), 
#  (0.5340642066240259, 'Pizza is a popular Italian food')
# ]


# RAG v1
knowledge_base = []
docs = [
    "React components are reusable UI building blocks.",
    "useState is a React Hook used to manage local state.",
    "useEffect is a React Hook that runs side effects after render.",
    "Next.js supports server-side rendering and routing.",
    "FastAPI is a Python framework for building APIs."
]

def get_embedding_from_ollama(doc):
    return ollama.embed(
        model="nomic-embed-text",
        input=doc
    )
for doc in docs:
    embedding = get_embedding_from_ollama(doc)
    knowledge_base.append({
        "text" : doc,
        "embedding" : embedding.embeddings[0]
    })

query_embedding = get_embedding_from_ollama("what is useEffect").embeddings[0]
results = []
for item in knowledge_base:
    score = cosine_similarity(query_embedding,item["embedding"])
    results.append((score,item["text"]))
results.sort(reverse=True)
top_docs = results[:2]
texts = [doc for score,doc in top_docs]
context = "\n\n".join(texts)


# RAG v2 - FAISS

#Create embedding matrix
# dtype means data type
# When NumPy creates an array, it wants to know: What kind of numbers am I storing?
# embeddings = np.array(
#     [ item["embedding"] for item in knowledge_base],dtype=np.float32
# )
#print(embeddings.shape) (5,768)

# create search engine or search struc
# IndexFlatL2 is also doing exact search-- linear but fast
# index = faiss.IndexFlatL2(768) #we are telling FAISS: "Every vector I will store has 768 dimensions."
# index.add(embeddings)

#[] around embedding because faiss wants Because FAISS expects: shape (number_of_queries, dimension)
# query_vector = np.array(
#     [get_embedding_from_ollama("what is useEffect").embeddings[0]],dtype=np.float32
# )
# 2 means return top_k matches or results
# D, I = index.search(query_vector,2) #O(n)
# top_docs = [docs[idx] for idx in I[0]]

# context = "\n\n".join(top_docs)
# prompt = f"""
# Answer the question using the context below.
# Context:
# {context}

# Question:
# "what is useEffect?"
# """
# response = ollama.chat(
#     model="llama3",
#     messages=[{
#         "role" : "user",
#         "content" : prompt
#     }]
# )
# answer = response["message"]["content"]

# production style rag
# with open("react_docs.txt","r",encoding="utf-8") as f:
#     document = f.read()
loader = TextLoader("react_docs.txt",encoding="utf-8")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size = 500,chunk_overlap = 100)
chunks = splitter.split_documents(documents=documents)
embedding_model = OllamaEmbeddings(
    model="nomic-embed-text"
)
query =  "What is state management?"
vector_store = FAISS.from_documents(chunks,embedding_model)
retriever = vector_store.as_retriever()
results = retriever.invoke(
   query
)
prompt_template = PromptTemplate(
    template="""
You are a React expert.

Use the following context to answer the question.

Context:
{context}

Question:
{question}

Answer:

""", input_variables=["context","question"]
)

context =  [ doc.page_content for doc in results]
context = "\n\n".join(context)

prompt = prompt_template.invoke({
    "context" : context,
    "question" : query
})
llm = ChatOllama(
    model="llama3:latest"
)
response = llm.invoke(prompt)
print(type(response))
print(response.content)
# paragraphs = document.split("\n\n")

# paragraph_embeddings = []
# for p in paragraphs:
#     embedding = get_embedding_from_ollama(p)
#     paragraph_embeddings.append(
#        embedding.embeddings[0]
#     )

# def chunk_text(document,chunk_size,overlap):
#     if overlap > chunk_size:
#         raise ValueError("overlap value must be less than chunk size")
#     overlapping_by = chunk_size - overlap
#     chunks = []
#     start_index = 0
#     while start_index < len(document):
#         chunk = document[start_index: start_index + chunk_size]
#         if overlap < chunk_size:
#             start_index += overlapping_by
#         chunks.append(chunk)
#     return chunks


# chunks_list = chunk_text(document,chunk_size=500,overlap=100)


# knowledge_basee = []


# for chunk in chunks_list:
#     embedding = get_embedding_from_ollama(chunk)
#     knowledge_basee.append({
#         "text" : chunk,
#         "embedding" : embedding.embeddings[0]
#     })

# new_embeddings = np.array(
#     [ item["embedding"] for item in knowledge_basee],dtype=np.float32
# )
# index = faiss.IndexFlatL2(768) #we are telling FAISS: "Every vector I will store has 768 dimensions."
# index.add(new_embeddings)
# query_vector = np.array(
#     [get_embedding_from_ollama("what is state management in react").embeddings[0]],dtype=np.float32
# )
# D, I = index.search(query_vector,2)
# print(D)
# print(I)
# top_docs = [knowledge_basee[idx]["text"] for idx in I[0]]
# # print(top_docs)
# # print(knowledge_basee[14]["text"])
# print(knowledge_basee[24]["text"])
