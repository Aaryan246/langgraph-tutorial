from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()

llm = ChatOllama(model="gemma4:e4b", num_ctx=16384, temperature=0)


class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: bool = Field(
        description="True if the answer addresses and resolves the question, False otherwise."
    )


structured_llm_grader = llm.with_structured_output(GradeAnswer, method="json_schema")

system = """You are a grader assessing whether an answer addresses and resolves a user question.
Respond ONLY with a valid JSON object matching this schema: {{"binary_score": true}} or {{"binary_score": false}}.
Use true if the answer resolves the question, and false if it does not."""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question:\n\n{question}\n\nLLM generation:\n\n{generation}"),
    ]
)

answer_grader: RunnableSequence = answer_prompt | structured_llm_grader
