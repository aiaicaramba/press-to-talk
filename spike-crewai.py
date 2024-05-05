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
from crewai import Task
from crewai import Crew, Process



ollama_model    = Ollama(model="phi3")


llm=ollama_model   # introduced to be able to switch easy between llm's.

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

#print("----------------------")
#print(project_template)
#print("----------------------")
#print(project_example)
#print("----------------------")



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
    allow_delegation=False,
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




# Research task
write_task  = Task(
    description=(
        "Write the challenge for students to markdown file. "
        f"Use this template: <<<{project_template}>>>, and example <<<{project_example}>>>"
    ),
    expected_output='A comprehensive challenge description.',
    tools=[],
    agent=writer,
    output_file="generated/proj_Gen.md" 
)


make_up_challenge_task = Task(
    description=(
        f"The interests and hobbies of a student are {interest_of_user}. Make up a Challenge for this student."
        f"A challenging and motivating Challenge for a student. Format the challenge exactly like this: <<<{project_template}>>>"
        
    ),
    expected_output="An  interesting challenge for a student.",
    tools=[],
    agent=mediator,
    async_execution=False,
    human_input=False,
)




# Forming the tech-focused crew with enhanced configurations
crew = Crew(
    agents=[mediator, writer],
    tasks=[make_up_challenge_task, write_task]
    ,
    process=Process.sequential  # Optional: Sequential task execution is default
)



# Starting the task execution process with enhanced feedback
result = crew.kickoff(inputs=
    {'theme': interest_of_user, 
     'layers': ['ICT & Software Engineering','ICT & Business', 'ICT & AI']
    })

print(result)



