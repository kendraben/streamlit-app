# Bring in deps
import os 
from apikey import apikey 

import streamlit as st 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.chains import ConversationChain
#from langchain.memory import ConversationBufferMemory
#from langchain.utilities import WikipediaAPIWrapper 

os.environ['OPENAI_API_KEY'] = apikey

st.markdown("# Single Framing")
#st.sidebar.markdown("# Verdi Concepts")

# App framework
#st.title('Concepts')
prompt1 = st.text_input('Describe your opportunity:') 

# Prompt templates
title_template = PromptTemplate(
    input_variables = ['opportunity'], 
    template='''With the following opportunity description: {opportunity}. With the opportunity description, list the most important 5-10 questions that should be answered to start framing out the opportunity. Return the 5-10 foundational questions only in your output, each question on its own line, similar to the example outputs listed below. 
    EXAMPLE Output 1: 
    Objective: What is the objective or goal?
    Stakeholders: Who are the stakeholders or target audience?
    Impact Scope: What is the market size or impact scope?
    Alternatives: What are the alternatives to doing this?
    Resources: What resources are required?
    Risks: What are the risks involved?
    Metrics: What metrics will measure success?
    Timeline: What is the timeline for implementation or completion?
    
    EXAMPLE Output 2:
    Outcome: What is the target outcome or goal?
    Target Customer: Who is the target audience?
    Impact Scope: What is the market size or impact scope?
    Competitors: Who are the competitors or alternatives?
    Resources: What resources are required?
    Risks: What are the risks involved?
    Metrics: What metrics will measure success?
    '''
)
#You have the following opportunity description: {opportunity}. With the opportunity description, you do the following: 1. determine what type of opportunity it is, 2. based on what you know of the opportunity, ask the most important 5-10 questions that should be answered to start framing out the opportunity. Format using the example given. 

script_template = PromptTemplate(
    input_variables = ['opportunity','questions'], 
    template='''Based on the opportunity description: {opportunity}, and the questions for this opportunity: {questions}, give your best estimation for the answers very concisely, with each answer on its own line, similar to the example output below.
    EXAMPLE Output:
    Objective: To establish thought leadership in the AI space. \n
    Stakeholders: AI enthusiasts, researchers, and the general public. \n
    Impact Scope: Global; anyone interested in AI. \n
    Competitors: Other AI blogs and publications. \n
    Resources: Content writers, SEO experts, and web hosting. \n
    Estimated Cost: $5,000 - $10,000. \n
    ROI: Increased website traffic and brand recognition. \n
    Risks: Low engagement, outdated or incorrect information. \n
    Metrics: Website traffic, engagement metrics, and subscriber count. \n
    Timeline: 3-6 months.
    '''
)

# Llms
# need the prompt chain to output responses into multiple variables and craft them dynamically if possible
llm = OpenAI(temperature=0.9) 
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='questions')
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='concepts')
sequentialchain = SequentialChain(chains=[title_chain,script_chain], input_variables=['opportunity'], output_variables=['concepts'], verbose=True)




if prompt1:
    response = sequentialchain(prompt1)
    #st.write(response)
    st.write(response['concepts'])