import chainlit as cl
from system_graph import builder, advanced_search_graph
from langchain.schema.runnable.config import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
import asyncio


memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

@cl.on_chat_start
async def on_chat_start():
    initial_state = {
        "user_query": None,
        "rewritten_query": None,
        "need_internet_search": None,
        "search_results": None,
        "bad_state": None,
        "final_answer": "",
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
    
    # Send an initial message with a loading indicator.
    loading_msg = cl.Message(content="🔍 Searching and reasoning...")
    await loading_msg.send()
    
    try:
        async for chunk in advanced_search_graph.astream(current_state, config=RunnableConfig(**config)):
            # Debug print to see the complete chunk structure.
            print("Chunk received:", chunk)
            
            # Only process tokens from the final node.
            if "node_previous_chat_summarizer" in chunk:
                final_node_state = chunk["node_previous_chat_summarizer"]
                token = final_node_state.get("final_answer", "")
                if token:
                    # Artificial delay so you can see streaming in action.
                    await asyncio.sleep(1)
                    await loading_msg.stream_token(token)
                    current_state["final_answer"] += token
                    cl.user_session.set("state", current_state)
    except Exception as e:
        await cl.Message(content=f"Error: {str(e)}").send()
        return

    # Finalize the message update.
    await loading_msg.update()