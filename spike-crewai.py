# Dependencies
# pip install -q crewai 'crewai[tools]'

# Credentials
import os
import getpass

#os.environ['OPENAI_API_KEY'] = getpass.getpass("OPENAI_API_KEY")
#os.environ["SERPER_API_KEY"] = getpass.getpass("SERPER_API_KEY") # serper.dev API key for Google Searches


from crewai import Agent
from crewai_tools import SerperDevTool

# ollama 
from langchain_community.llms import Ollama
ollama_model    = Ollama(model="phi3")

#search_tool = SerperDevTool()



def ReadEntireFile(file_name):
    with open(file_name, "r") as file:
        data = file.read()
    return data
# -----------User Q & A
# --------------------------------------------------------------

print("Welcome to the Fontys ICT sem1 Challenge Generator!")
interest_of_user = input("What are your interests and hobbies? ")
print(f"Your interests are: {interest_of_user}")


# -----------    Templates and examples.
# --------------------------------------------------------------

project_template = ReadEntireFile("proj_Template.md")
project_example = ReadEntireFile("proj_Example.md")

print("----------------------")
print(project_template)
print("----------------------")
print(project_example)
print("----------------------")



# Creating a senior researcher agent with memory 
# --------------------------------------------------------------
mediator = Agent(
    role='Ideator',
    goal='Make up challenging, broad Challenges for students to orient in the broad field of ICT.',
    verbose=True,
    memory=True,
    backstory=(
        "Make it exciting and challenging for students to orient in the broad field of ICT."
        "Some students already have an idea what they want to do, others don't. Also, some have ideas about a theme they want to work in, and it wouldd be nice if they can dive into that. But it's also nice to challenge them to think outside the box."
        "When the student does not have an idea, maybe ask about their hobbies or plans for the future, or why they chose this study, and invite them to find a theme that motivates them."
        "It is nice if you can make up a challenge that invites to think about ICT in the most broad sense!"
    ),
    tools=[],
    allow_delegation=True,
    llm=ollama_model
)

# Creating a writer agent with custom tools and delegation capability
writer = Agent(
    role='Writer',
    goal='Write a tailor-made orienting ICT challenge for a Fontys ICT semester1-student.',
    verbose=True,
    memory=True,
    backstory=(
        "With a flair for communicating to students, young people of mostly 17-25 years old, "
        "your task is to create, together with a student, a tailor-made orienting ICT challenge, for a Fontys ICT semester1-student."
        "Being a Dutch HBO (Hoger Beroeps Onderwijs) for ICT, we adhere to the HBO-i-framework, which you can find at https://www.hbo-i.nl ."
    ),
    tools=[],
    allow_delegation=False,
    llm=ollama_model
)



from crewai import Task

# Research task
make_up_challenge_task  = Task(
    description=(
        "Make up a broad Challenge for students to orient in the broad field of ICT."

    ),
    expected_output='A comprehensive challenge description.',
    tools=[],
    agent=mediator,
)

# Writing task with language model configuration
write_task = Task(
    description=(
        "Write an insightful challenge on <<<{theme}>>>."
        "Focus on current trends."
        "This challenge should be easy to understand, engaging, motivating to learn new technology."
        f"Use template: <<<{project_template}>>> and example <<<{project_example}>>>."
    ),
    expected_output=f"The challenge written in markdown, using this template: <<<{project_template}>>>",
    tools=[],
    agent=writer,
    async_execution=False,
    output_file="generated/proj_Gen.md" 
)

ask_for_interests_task = Task(
    description=(
        "Discuss with the student their interests and hobbies, and try to find a theme that motivates them."
    ),
    expected_output='A theme that motivates the student.',
    tools=[],
    agent=mediator,
    async_execution=False,
    human_input=True
)


from crewai import Crew, Process


# Forming the tech-focused crew with enhanced configurations
crew = Crew(
    agents=[mediator, writer],
    tasks=[ask_for_interests_task , make_up_challenge_task, write_task]
    ,
    process=Process.sequential  # Optional: Sequential task execution is default
)



# Starting the task execution process with enhanced feedback
result = crew.kickoff(inputs=
    {'theme': interest_of_user, 
     'layers': ['ICT & Software Engineering','ICT & Business', 'ICT & AI']})
print(result)



