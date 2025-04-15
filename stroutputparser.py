from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Define the Hugging Face model
llm = HuggingFaceEndpoint(repo_id="google/gemma-2-2b-it", task="text-generation")

# Create the chat model
model = ChatHuggingFace(llm=llm)

# 1st prompt - detailed report
template1 = PromptTemplate(
    template="write a detailed report on {topic}", input_variable=["topic"]
)

# 2nd prompt - summary

template2 = PromptTemplate(
    template="write a 5 line summary on the following text. /n {text}",
    input_variable=["text"],
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({"topic": "black hole"})

print(result)
