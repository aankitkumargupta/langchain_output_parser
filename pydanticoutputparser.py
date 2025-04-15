from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


# Load environment variables
load_dotenv()

# Define the Hugging Face model
llm = HuggingFaceEndpoint(repo_id="google/gemma-2-2b-it", task="text-generation")

# Create the chat model
model = ChatHuggingFace(llm=llm)


class City(BaseModel):
    name: str = Field(description="Name of the mayor")
    population: int = Field(description="Population of the city")
    area: int = Field(description="area of the city in sq meters")
    country: str = Field(description="Name of the country where the city is situated")


parser = PydanticOutputParser(pydantic_object=City)

template = PromptTemplate(
    template="Return the name, population and country of a {city} \n {format_instruction}",
    input_variables=["city"],
    partial_variables={"format_instruction": parser.get_format_instructions()},
)

# prompt = template.invoke({"city": "Mumbai"})

# result = model.invoke(prompt)

# final_result = parser.parse(result.content)

chain = template | model | parser

final_result = chain.invoke({"city": "Mumbai"})


print(final_result)
