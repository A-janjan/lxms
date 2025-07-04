from langgraph.prebuilt import create_react_agent
from vertexai import init


init(project="using-gemini-after26jun25", location="us-central1")



def get_weather(city: str) -> str :
    """Get weather for a given city."""
    return f"The weather in {city} is sunny with a temperature of 25°C."


agent = create_react_agent(
    model="gemini-2.5-flash-preview-05-20",
    tools=[get_weather],
    prompt="you are a helpful assistant that provides weather information.",
)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)


print(response)
