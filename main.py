from dotenv import load_dotenv
from typing import List, Sequence
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import MessagesState, StateGraph, END
from chains import reflection_chain, generate_chain

load_dotenv()

REFLECT = "reflect"
GENERATE = "generate"


def generation_node(state: MessagesState):
    res = generate_chain.invoke({"messages": state["messages"]})
    return {"messages": [res]}


def reflection_node(state: MessagesState):
    res = reflection_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=res.content)]}

builder = StateGraph(MessagesState)

builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)


def should_continue(state: MessagesState):
    if len(state["messages"]) > 4:
        return END
    return REFLECT

builder.add_conditional_edges(GENERATE, should_continue, {END: END, REFLECT: REFLECT})

builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()

#UNCOMMENT TO GET A MERMAID CODE TO SEE VISUAL ON MERMAID SITE
# print(graph.get_graph().draw_mermaid())

if __name__ == "__main__":
    print("Hello LangGraph")
    inputs = {
        "messages": [
            HumanMessage(
                content="""Make this tweet better:"
                                    @LangChainAI
            — newly Tool Calling feature is seriously underrated.

            After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.

            Made a video covering their newest blog post

                                  """
            )
        ]
    }
    response = graph.invoke(inputs)
    print(response)

