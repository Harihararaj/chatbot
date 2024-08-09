import vertexai
import streamlit as st
from vertexai.preview import generative_models
from vertexai.preview.generative_models import GenerativeModel, Part, Content, ChatSession

def stream_data_from_llm(chat : ChatSession, query: str):
    '''
    Method to stream data from the llm.
    @chat: VertexAI llm ChatSession.
    @query: User provided prompt.
    '''
    for partial_response in chat.send_message(query, stream = True):
        yield partial_response.text

def chat_with_llm(chat : ChatSession, prompt : str):
    '''
    Get the prompt from the user, generate response from the llm and stream that generated data to the user via Streamlit UI.
    @chat: VertexAI llm ChatSession.
    @prompt: User provided prompt.
    '''
    try:
        llm_response = st.chat_message(ASSISTANT).write_stream(stream_data_from_llm(chat, prompt))
        st.session_state[MESSAGES].append(
            {
                ROLE : USER,
                CONTENT : prompt
            }
        )
        st.session_state[MESSAGES].append(
            {
                ROLE : MODEL,
                CONTENT : llm_response
            }
        )
    except Exception as e:
        st.error(f"An error occurred during streaming: {e}")

project = "sample-gemini-424220"
vertexai.init(project = project)
config = generative_models.GenerationConfig(
    temperature=0.0,
    max_output_tokens=100                         
)
model = GenerativeModel(
    "gemini-pro",
    generation_config = config
)
chat = model.start_chat()

USER = 'user'
ASSISTANT = 'ai'
MESSAGES = 'messages'
ROLE = 'role'
CONTENT = 'content'
MODEL = 'model'

if MESSAGES not in st.session_state:
    st.session_state[MESSAGES] = []

for index, message in enumerate(st.session_state[MESSAGES]):
    content = Content(
        role = message[ROLE],
        parts = [Part.from_text(message[CONTENT])]
    )

    chat.history.append(content)

    if index!=0:
        if message[ROLE] == USER:
            st.chat_message(USER).write(message[CONTENT])
        else:
            st.chat_message(ASSISTANT).write(message[CONTENT])

prompt : str = st.chat_input("Enter a prompt here")
if prompt:
    st.chat_message(USER).write(prompt)
    chat_with_llm(chat, prompt)

if len(st.session_state.messages) == 0:
    initial_prompt = "Introduce yourself as ReX, an assistant powered by Google Gemini. You use emojis to be interactive. Also the \
    introduction should be only for 2 lines"
    chat_with_llm(chat, initial_prompt)

