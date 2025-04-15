from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Load environment variables
load_dotenv()

# Define the Hugging Face model
llm = HuggingFaceEndpoint(repo_id="google/gemma-2-2b-it", task="text-generation")

# Create the chat model
model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

# 1st prompt - detailed report
template = PromptTemplate(
    template="Provide me the land area, population, longest river, president name and national animal of India \n {format_instruction}",
    input_variables=[],
    partial_variables={
        "format_instruction": parser.get_format_instructions()
    },  # filled before runtime
)

# prompt = template.format()

# result = model.invoke(prompt)

# final_result = parser.parse(result.content)


# using chain

chain = template | model | parser

result = chain.invoke({})

print(result)
