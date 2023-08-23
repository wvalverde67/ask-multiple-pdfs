import time
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import create_tagging_chain_pydantic
from langchain.chat_models import ChatOpenAI
from pydantic import BaseModel, Field, conlist
from typing import Optional
import plotly.express as px
import pandas as pd


# Using the Pydantic schema to specify the required properties of our desired candidates’ labels. 
# Create a new object PersonalDetails derived from Pydantic BaseModel. 
# Make sure every property has the same name as the element in the ask_for list.
class PersonalDetails(BaseModel):
    full_name: Optional[str] = Field(
        None,
        description="Es el nombre completo del usuario.",
    )
    school_background: Optional[conlist(int, min_items=3, max_items=3)] = Field(
        None,
        description="""Nivel de cualificación del historial educativo. El rango es de 1 a 10, siendo el número mayor el de mayor cualificación.
        El primer elemento indica el nivel de título, 10 significa maestría o superior, 1 significa secundaria.
        El segundo elemento indica la relevancia de la especialidad, 10 significa ciencias de la computación y carreras relacionadas.
        El tercer elemento indica el ranking de la universidad, 10 significa estar entre las 50 mejores universidades del mundo, 1 significa universidad comunitaria.
        0 significa indeterminado.
        """,
    )
    working_experience: Optional[conlist(int, min_items=3, max_items=3)] = Field(
        None,
        description="""Estado de cualificación del historial profesional. El rango es de 1 a 10, siendo el número mayor el de mayor cualificación.
        El primer elemento indica el nivel del puesto, 10 significa gerente senior o superior, 1 significa pasante.
        El segundo elemento indica la relevancia del puesto, 10 significa puestos de desarrollo de software.
        El tercer elemento indica el ranking de la empresa, 10 significa estar entre las 500 mejores empresas del mundo, 1 significa pequeña empresa local.
        0 significa indeterminado.
        """,

    )
    interview_motivation: Optional[int] = Field(
        None,
        description="""El nivel de motivación del candidato para asistir a la entrevista.
        10 significa muy interesado y entusiasmado con la entrevista y la nueva oportunidad laboral. 1 significa desinteresado.
        """,
    )

def ask_for_info(ask_for, llm):

    prompt = ChatPromptTemplate.from_template(
        """Eres un reclutador de empleo que solo hace preguntas.
        Lo que estás pidiendo son y solo deben estar en la lista de "ask_for".
        Después de seleccionar un elemento en la lista de "ask_for", debes extenderlo con 20 palabras más en tus preguntas con más pensamientos y guía.
        Solo debes hacer una pregunta a la vez, incluso si no obtienes todo de acuerdo con la lista de "ask_for".
        ¡No preguntes como si fuera una lista!
        Espera las respuestas del usuario después de cada pregunta. No inventes respuestas.
        Si la lista de "ask_for" está vacía, entonces agradéceles y pregúntales cómo puedes ayudarles.
        No saludes ni digas hola.
        ### lista de "ask_for": {ask_for}
        """
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    question = chain.run(ask_for=ask_for)

    return question


# check_what_is_empty() returns a new ask_for list with empty items in the PersonalDetails type of input.
def check_what_is_empty(user_peronal_details):
    ask_for = []
    for field, value in user_peronal_details.dict().items():
        if value in [None, "", 0]:  
            print(f"Field '{field}' is empty.")
            ask_for.append(f'{field}')
    return ask_for

# returns a new PersonalDetails type of instance with new and old details combined.
# Please note that “empty” means either of [None, “”, 0] according to various data types.
def add_non_empty_details(current_details: PersonalDetails, new_details: PersonalDetails):
    non_empty_details = {k: v for k, v in new_details.dict().items() if v not in [None, "", 0]}
    updated_details = current_details.copy(update=non_empty_details)
    return updated_details


# filter_response() to process the user’s responses (candidate’s answers).
def filter_response(text_input, user_details, tagging_chain):
    res = tagging_chain.run(text_input)
    user_details = add_non_empty_details(user_details,res)
    ask_for = check_what_is_empty(user_details)
    return user_details, ask_for

def radar_chart(motivation, education, career):
    df = pd.DataFrame(dict(
    r=[motivation,
       education[0],
       education[1],
       education[2],
       career[0],
       career[1],
       career[2]
       ],
    theta=['Motivación', 'Escolaridad','Grado Académico','Ranking Universitario',
           'Nivel Trabajo', 'Posición', 'Ranking Compañía']))

    fig = px.line_polar(df, r='r', theta='theta',  line_close=True,
                    color_discrete_sequence=px.colors.sequential.Plasma_r,
                    template="plotly_dark", title="Candidate's Job Match", range_r=[0,10])
    st.sidebar.header('Información solo Reclutador:')
    st.sidebar.write(fig)

def main():  # sourcery skip: extract-method
    load_dotenv()

    # Define the element of the ask_for list, as initially, we create an ask_init object.
    ask_init = ['full_name', 'school_background', 'working_experience', 'interview_motivation']

    # define the llm model
    llm = ChatOpenAI(temperature=0.5, model="gpt-3.5-turbo-0613")

    # create the tagging chain
    tagging_chain = create_tagging_chain_pydantic(PersonalDetails, llm)

    # define the initial user details
    user_init_bio = PersonalDetails(full_name="",
                                school_background=None,
                                working_experience=None,
                                interview_motivation=0)


    if "messages" not in st.session_state:
        question = ask_for_info(ask_init, llm)
        st.session_state.messages = [{"role":"assistant", "content":question}]
    if "details" not in st.session_state:
        st.session_state.details = user_init_bio
    if "ask_for" not in st.session_state:
        st.session_state.ask_for = ask_init
    
    # Show all the history chat messages on the web page:

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if answer := st.chat_input("Por favor responda la pregunta. "):
    # Add user message to chat history
    
        st.session_state.messages.append({"role": "user", "content": answer})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(answer)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            st.session_state.details, st.session_state.ask_for = filter_response(answer, st.session_state.details, tagging_chain)
            if st.session_state.ask_for != []:
                assistant_response = ask_for_info(st.session_state.ask_for, llm)
            else:
                assistant_response = """Gracias por participar en esta entrevista.
                                        Le notificaremos los próximos pasos una vez que hayamos llegado a una conclusión.
                                     """

                final_details = st.session_state.details.dict()
                radar_chart(final_details['interview_motivation'], final_details['school_background'], final_details['working_experience'])

            for chunk in assistant_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == '__main__':
    main()