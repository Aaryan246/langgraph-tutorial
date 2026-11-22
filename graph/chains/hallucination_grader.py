from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()

llm = ChatOllama(model="gemma4:e4b", num_ctx=16384, temperature=0)


class GradeHallucinations(BaseModel):
    """Binary score for hallucinations present in generation answer."""

    binary_score: bool = Field(
        description="True if the answer is grounded in the facts, False otherwise."
    )


structured_llm_grader = llm.with_structured_output(
    GradeHallucinations, method="json_schema"
)

system = """You are a grader assessing whether an LLM generation is grounded in and supported by a set of retrieved facts.
Respond ONLY with a valid JSON object matching this schema: {{"binary_score": true}} or {{"binary_score": false}}.
Use true if the answer is grounded in the facts, and false if it is not."""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts:\n\n{documents}\n\nLLM generation:\n\n{generation}"),
    ]
)

hallucination_grader: RunnableSequence = hallucination_prompt | structured_llm_grader
