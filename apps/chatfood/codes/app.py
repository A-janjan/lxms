import chainlit as cl
from system_graph import builder
from langchain.schema.runnable.config import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver

# Compile your LangGraph and initialize memory.
memory = MemorySaver()
advanced_search_graph = builder.compile()

@cl.on_chat_start
async def on_chat_start():
    initial_state = {
        "user_query": None,
        "rewritten_query": None,
        "need_internet_search": None,
        "search_results": None,
        "final_answer": "",
        "bad_state": None,
        "previous_chats_summary": None,
    }
    cl.user_session.set("state", initial_state)
    await cl.Message(content="Hello! I am ChatFood. How can I help you?").send()

@cl.on_message
async def on_message(message: cl.Message):
    config = {"configurable": {"thread_id": cl.context.session.id}}
    current_state = cl.user_session.get("state")
    current_state["user_query"] = message.content
    current_state["final_answer"] = ""
    
    # Send an initial message indicating processing.
    loading_msg = await cl.Message(content="🔍 Searching and reasoning...").send()
    
    try:
        # Get the final processed state from the graph.
        final_state = await advanced_search_graph.ainvoke(
            input=current_state, config=RunnableConfig(**config)
        )

        # Extract and send the final answer.
        final_answer = final_state.get("final_answer", "No response generated.")
        await cl.Message(content=final_answer).send()

        # Update session state.
        current_state["final_answer"] = final_answer
        cl.user_session.set("state", current_state)

    except Exception as e:
        await cl.Message(content=f"Error: {str(e)}").send()