from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from todoist_api_python.api import TodoistAPI

load_dotenv()
todoist_api_key = os.getenv("TODOIST_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")
project_id = os.getenv("PROJECT_ID")

todoist = TodoistAPI(todoist_api_key)


@tool
def add_task(task, description=None):
    """Add a new task to the user's tasks list. Use this when user wants to add or create a task"""
    todoist.add_task(content=task, description=description)

@tool
def show_tasks():
    """Show all the tasks from todoist app. Use this when user wants to show all tasks as bulleted list"""
    tasks = []
    # If you want to filter by project, use the project_id parameter
    dump = todoist.get_tasks(project_id=project_id)
    #dump = todoist.get_tasks()
    for item in dump:
        for task in item:
            tasks.append(task.content)
    return tasks


llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    google_api_key=gemini_api_key,
    temperature=0.3
)
system_prompt = "You are a helpful assistant that can help with both task management and general questions. " \
                "For task management, you can: - Add tasks to the user's Todoist list - " \
                "You have access to the add_task tool for this purpose. " \
                "you can: - Show all tasks to the user - You have access to the show_tasks tool for this purpose" \
                "For general questions, " \
                "you can provide helpful information and answers. " \
                "Always be friendly and helpful in your responses"

tools = [add_task, show_tasks]

prompt = ChatPromptTemplate([
    ("system", system_prompt),
    MessagesPlaceholder("history"),
    ("user", "{input}"),
    MessagesPlaceholder("agent_scratchpad")
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent,
                               tools=tools,
                               verbose=False)
history = []

while True:
    user_input = input("You[Please say 'end' to end the chat]: ")
    if user_input == 'end':
        print("Have a good day! Bye!")
        break
    response = agent_executor.invoke({"input": user_input, "history": history})
    print(response["output"])
    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=response["output"]))


