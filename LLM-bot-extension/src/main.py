from typing import Union
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from transformers import AutoTokenizer, AutoModelForCausalLM
from torch import cuda

from uuid import uuid4
from pydantic import BaseModel

import requests
import time

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
model_id = "codellama/CodeLlama-7b-Instruct-hf"
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
#test
tokenizer = ""
model = ""

save_dir = "/models"

historique = {}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    #allow_origins=["https://github.com"],
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class PromtRequest(BaseModel):
    id: str
    promt: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/premierdem")
def premier_demarrage():
    global model
    global tokenizer

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    tokenizer.save_pretrained(save_dir)
    model.save_pretrained(save_dir)
    print(device)
    return {"Page": "Premier demarrage"}

# marche pas
@app.get("/demarrage")
def demarrage_volume():
    global model
    global tokenizer

    model = AutoModelForCausalLM.from_pretrained("./models/")
    tokenizer = AutoTokenizer.from_pretrained("./models/")

    return {"Page": "demarrage"}


@app.get("/connexion")
def create_session():
    session = uuid4()
    global historique

    #historique[session] = list[tuple[str, str]]

    return {"id": session}


@app.get("/deconnexion")
def deconnexion():
    print('deconnexion')


@app.post("/generate")
def generate(request: PromtRequest):
    start_time = time.time()
    prompt = f"<s>[INST] {request.promt.strip()} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

    output = model.generate(
        **inputs,
        max_new_tokens=500,
        eos_token_id=int(tokenizer.convert_tokens_to_ids('.'))
    )
    output = output[0].to(device)

    print(tokenizer.decode(output))
    print('--- %s secondes ---' % (time.time() - start_time))
    return {"result": tokenizer.decode(output)}


@app.get("/gen")
def test_generate():
    start_time = time.time()
    user = """
    Here is an example of python code:

    "
    def op_sum(x,y):
    	return 'Hello'
    "

    Here is a comment made for this code:

    "The function does not return an addition."

    Is this a relevant and respectful review comment for this code?
    """
    prompt = f"<s>[INST] {user.strip()} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

    output = model.generate(
        **inputs,
        max_new_tokens=200,
        eos_token_id=int(tokenizer.convert_tokens_to_ids('.'))
    )
    output = output[0].to(device)

    print(tokenizer.decode(output))
    print('--- %s secondes ---' % (time.time() - start_time))
    return {"result": tokenizer.decode(output)}

@app.get("/pr-infos")
async def get_pr_infos():

    # modifier le token, le nom du repo et le nombre de prs qu'on veut recueillir
    # exemple ici avec le repo tinygrad

    GRAPHQL_API_URL = 'https://api.github.com/graphql'
    HEADERS = {'Authorization': f'Bearer GITHUB_TOKEN'}

    query = f"""
        query {{
            search(type: ISSUE, query: "repo:tinygrad/tinygrad is:pr", first: 10) {{ 
                nodes {{
                ... on PullRequest {{
                    id
                    title
                    url
                    mergedAt
                    createdAt
                    closedAt
                    number
                }}
                }}
            }}
        }}
    """
    response = requests.post(GRAPHQL_API_URL, json={"query": query}, headers=HEADERS)
    if response.ok:
        data = response.json()
        prs = []
        for pr in data['data']['search']['nodes']:
            prs.append((pr['id'], pr['title'], pr['mergedAt'], pr['createdAt'], pr['closedAt'], pr['number']))
        return {"data": prs}
    else:
        raise RuntimeError(f"Query failed to run by returning code of {response.status_code}.")
    
class PromptMessage(BaseModel):
    prompt: str
    num_tokens: int

# To faciliate testing, here's an endpoint to test the LLM included.
# JSON Format should be 
# {
#   "prompt": "scenario template",
#   "num_tokens": 200 or the value you want
# }
# Here's a few scenarios exemple: https://docs.google.com/document/d/1OKnzy3pTW6oRd3671XEzIRW34GuDjbHlaVjotqIt6yA/edit
# If you are having trouble formatting the json, paste the scenario into the template and ask ChatGPT ;)
@app.post("/generate-prompt")
def message_generate(request: PromptMessage):
    # Ensure the tokenizer and model are already loaded
    if tokenizer == "" or model == "":
        raise HTTPException(status_code=503, detail="Model is not loaded")

    start_time = time.time()
    user_prompt = request.prompt.strip()
    prompt = f"<s>[INST] {user_prompt} [/INST]"

    # Prepare the prompt for the model
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

    # Generate the response using the specified number of tokens
    output = model.generate(
        **inputs,
        max_new_tokens=request.num_tokens,  # Use the specified number of tokens
        eos_token_id=int(tokenizer.convert_tokens_to_ids('.'))
    )
    output = output[0].to(device)

    generated_text = tokenizer.decode(output)

    print(f"--- {(time.time() - start_time)} seconds ---")
    return {"result": generated_text}