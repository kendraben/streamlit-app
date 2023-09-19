import os
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 

apiKey = st.secrets['OPENAI_API_KEY']

st.markdown("# Multiple Framings")

# App framework
#st.title('🦜🔗 Multi Version Prompt tester')
prompt = st.text_input('Describe your opportunity:') 

best_responses = {}  # Empty dictionary to store the best responses

llm = OpenAI(temperature=0.9) 

# Title Template
title_template = PromptTemplate(
    input_variables = ['opportunity'], 
    template='''With the following opportunity description: {opportunity}, list the most important 5-10 questions that should be answered to start framing out the opportunity. Return the 5-10 foundational questions only in your output, each question on its own line, similar to the example outputs listed below. 
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
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='response')

# Run title chain once
if prompt:
    single_response = title_chain.run(opportunity=prompt)
    #st.write(f"Title Response: {single_response}")

# Script Template
script_template = PromptTemplate(
    input_variables = ['opportunity','response'], 
    template='''Based on the opportunity description: {opportunity}, and the questions for this opportunity: {response}, give your expert opinion for the answers very concisely, with each answer on its own line, similar to the example output below.
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

# Run the script chain multiple times, each time with the same title response
for i in range(1, 4):
    script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key=str(i))
    
    if single_response:
        best_response = script_chain.run(opportunity=prompt, response=single_response)
        best_responses[i] = best_response  # Store each response in the dictionary

# Display the best responses
st.write(best_responses)
