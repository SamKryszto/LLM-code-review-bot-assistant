from typing import Union
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from transformers import AutoTokenizer, AutoModelForCausalLM
from torch import cuda

from uuid import uuid4
from pydantic import BaseModel

import time

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
model_id = "codellama/CodeLlama-7b-Instruct-hf"
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
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

    GRAPHQL_API_URL = 'https://api.github.com/graphql'
    HEADERS = {'Authorization': f'Bearer "GITHUB_ACCESS_TOKEN"'}

    query = f"""
        query {{}
            search(type: ISSUE, query: "repo:tinygrad/tinygrad is:pr", first: 100) {{
                nodes {{
                ... on PullRequest {{
                    id
                    title
                    url
                    mergedAt
                    createdAt
                    number
                }}
                }}
            }]
        }}
    """

    if resp.status_code == 200:
        data = await resp.json()
        org_name = data['data']['organization']['name']
        project_title = data['data']['organization']['projectV2']['title']
        prs = []
        for repo in data['data']['organization']['projectV2']['repositories']['nodes']:
            for pr in repo['pullRequests']['nodes']:
                prs.append((pr['id'], pr['title'], pr['createdAt'], pr['closedAt']))
        return {"data": {"organization": org_name, "project": project_title, "prs": prs}}
    else:
        raise RuntimeError(f"GraphQL API returned status code {resp.status_code}: {await resp.text()}")


    print("pr infos!")
