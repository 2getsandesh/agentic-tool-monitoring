import os, types, time, random
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames
from ibm_watsonx_ai.foundation_models import ModelInference
from pprint import pprint
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow, tool
from langchain_ibm import WatsonxLLM

from dotenv import load_dotenv
load_dotenv()

Traceloop.init(app_name="watsonx_chat_service")

def watsonx_llm_init() -> ModelInference:
    watsonx_llm_parameters = {
        GenTextParamsMetaNames.DECODING_METHOD: "sample",
        GenTextParamsMetaNames.MAX_NEW_TOKENS: 100,
        GenTextParamsMetaNames.MIN_NEW_TOKENS: 1,
        GenTextParamsMetaNames.TEMPERATURE: 0.5,
        GenTextParamsMetaNames.TOP_K: 50,
        GenTextParamsMetaNames.TOP_P: 1,
    }
    models = ['ibm/granite-3-2-8b-instruct']
    model = random.choice(models)
    watsonx_llm = WatsonxLLM(
        model_id=model,
        url=os.getenv("WATSONX_URL"),
        apikey=os.getenv("WATSONX_API_KEY"),
        project_id=os.getenv("WATSONX_PROJECT_ID"),
        params=watsonx_llm_parameters,
    )
    return watsonx_llm

@tool(name="watsonx_llm_langchain_question_123")
def tool_1():
    return "Hi tool 1"

@workflow(name="watsonx_llm_langchain_question")
def watsonx_llm_generate(question):
    watsonx_llm = watsonx_llm_init()
    return watsonx_llm.invoke(question)

for i in range(2):
    question_multiple_responses = [ "What is AIOps?", "What is GitOps?"]
    question = random.choice(question_multiple_responses)
    response = watsonx_llm_generate(question)
    tool_1()
    if isinstance(response, types.GeneratorType):
        for chunk in response:
            print(chunk, end='')
    pprint(response)
    time.sleep(3)