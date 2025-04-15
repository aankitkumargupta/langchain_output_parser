from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

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
    template="write a 5 pointer summary on the following text. /n {text}",
    input_variable=["text"],
)

prompt1 = template1.invoke({"topic": "black hole"})

result = model.invoke(prompt1)

prompt2 = template2.invoke({"text": result.content})

result1 = model.invoke(prompt2)

print(result1.content)
