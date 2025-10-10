import chromadb
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter

import streamlit as st
import os
import json
from utils.chroma_rag import RAG



st.markdown(
    """
# Document RAG App
Ask Questions about Your Documents
"""
)
file_path = "data/sections.json"
if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
    with open(file_path, "r", encoding="utf-8") as f:
        section_dict = json.load(f)
else:
    section_dict = {}  # default to empty list if file is empty


if "sections" not in st.session_state:
  st.session_state.sections = section_dict

def save_session_to_json():
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(st.session_state.sections, f, indent=4)

# data = st.file_uploader("Upload Your File", type=['pdf','txt'])


# all of this code structures the side bar, allowing the user to add different sections for their formating

# api_key = os.getenv("OPENAI_API_KEY")

# if api_key == None:
#     api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# os.environ["OPENAI_API_KEY"] = api_key

# if api_key !=None:
rag = RAG()

# st.write(api_key)

# AI model instance & vector db?


# adding the topics groups
with st.sidebar.expander("Adding Topics"):
    st.markdown("Create your document directory")
    new_topic = st.text_input("enter a new topic")
    new_topic = new_topic.title()
    if st.button("Add Category") and new_topic:
        if new_topic not in st.session_state.sections.keys():
            st.session_state.sections[new_topic] = {}
            save_session_to_json()
        else:
            st.warning("That topic already exists")
# adding sections
    with st.expander("Adding Sections"):
        if len(st.session_state.sections) > 0:
            topic = st.selectbox("Choose a topic to add a section to", st.session_state.sections.keys())

            new_section = st.text_input("Add a section to your topic")
            new_section = new_section.title()

            if st.button("Add Section") and new_section:
                if new_section not in st.session_state.sections[topic].keys():
                    st.session_state.sections[topic][new_section] = []
                    save_session_to_json()
                else:
                    st.warning("That section already exists")
# addng the subsections
    with st.expander("Add a subsection"):
        if len(st.session_state.sections) > 0:
            topic = st.selectbox("Choose a topic to add a section to", st.session_state.sections.keys(), key='topic_select')
            if topic:
                sections = st.session_state.sections[topic]
                if len(sections.keys()) > 0 :
                    section = st.selectbox("Choose Sections", sections.keys(), key='section_select')
                    new_subsection = st.text_input("Add a subsection")
                    new_subsection = new_subsection.title()

                    if st.button("Add Sub-Section") and new_subsection:
                        if new_subsection not in st.session_state.sections[topic][section]:
                            st.session_state.sections[topic][section].append(new_subsection)
                            save_session_to_json()
                        else:
                            st.warning("That section already exists")

# adding documents
with st.sidebar.expander("Add Documents"):
    st.write("Select Location of Document")
    if len(st.session_state.sections) > 0:
            file_pth = ""
            query_topic = st.selectbox("Choose a topic to add a section to", st.session_state.sections.keys(), key='topic_select_doc')
            if query_topic:
                file_pth += f"\{query_topic} "
                query_sections = st.session_state.sections[query_topic]
                if len(query_sections.keys()) > 0 :
                    query_section = st.selectbox("Choose Sections", query_sections.keys(), key='section_select_doc')
                    query_subsections = query_sections[query_section]
                    file_pth += f"\{query_section} "

                    if len(query_subsections) >= 0:
                        query_subsection = st.multiselect("Choose Sections", query_subsections, key='subsection_select_doc')
                        file_pth += f"\{query_subsection} "

                st.write(file_pth)
                data = st.file_uploader("Upload Your File", type=['pdf','txt'])
                if data:
                    filename = data.name
                    clean_filename, ext = os.path.splitext(filename)
                    if ext == ".pdf":
                        text = rag.pdf_reader(data)
                        rag.adding_chunked(full_text=text, doc_name= clean_filename, topic=query_topic, section=query_section, subsection=query_subsection[0])
                    st.write(clean_filename)
                # after data is entered, you basically add to your vector database, with the file information
                # for meta data.
                # we will create a seperate function in utils that we can call for ease of dev
                # this is where we will introduce the vector_database/ edit and update?


with st.sidebar.expander("Choosing Sections for Query"):
    if len(st.session_state.sections) > 0:
            query_topic = st.selectbox("Choose a topic to add a section to", st.session_state.sections.keys(), key='topic_select_query')
            if query_topic:
                query_sections = st.session_state.sections[query_topic]
                if len(query_sections.keys()) > 0 :
                    query_section = st.selectbox("Choose Sections", query_sections.keys(), key='section_select_query')
                    query_subsections = query_sections[query_section]

                    if len(query_subsections) >= 0:
                        query_subsection = st.multiselect("Choose Sections", query_subsections, key='subsection_select_query')
                        
                    
chat_c = st.container()
chat_c.markdown("## Ask Questions About your Documents:")
chat_c.markdown("Subsections and Tags:")
path_str = ""

if query_topic:
    path_str += f":blue-badge[{query_topic}] "
    if query_section:
        path_str += f":blue-badge[{query_section}] "
        if query_subsection:
            for i in range(len(query_subsection)):
                path_str += f":primary-badge[{query_subsection[i]}] "

chat_c.markdown(path_str)


# basically we will use the tags as a filter, limiting what the chatbots has access to for their rag template
chat_txt_block = chat_c.container(height=400)

chatbot_c = chat_c.container()

# saving chat into the session & display
if "messages" not in st.session_state:
    st.session_state.messages = rag.messages

for message in st.session_state.messages:
    with chat_txt_block.chat_message(message["role"]):
        chat_txt_block.markdown(message["content"])

prompt = chat_c.chat_input("Say something")

if prompt:
    # users input
    chat_txt_block.chat_message("user", avatar="😯").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # function for assitants output
    # this is where we would replace the response wth our generated answer
    # response = f"Echo: {prompt}"
    response = rag.ask(prompt, topic=query_topic, section=query_section)
    #response = ask(prompt, topic=query_topic, section=query_section, subsection=query_subsection)
# 😎, 
    with chat_txt_block.chat_message("assistant", avatar="😎"):
        chat_txt_block.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

# we will use the st.session_state.messages section to create our conversational context.
# st.write(st.session_state.messages)