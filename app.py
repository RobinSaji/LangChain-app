import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS#FAISS is for storring the result , it will store the embediing in our local machine ,if we close the application then the entire emeddings will be gone
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from htmlTemplates import css, bot_template,user_template

#creating a funcion to extract the text from the pdf 
def get_pdf_text(pdf_docs):
    #a variable to store all the text in the pdf by concatenating
    text =""
    for pdf in pdf_docs:
        #we create a pdfreader object for each pdfs which has pages,  it is actually the pages that you are able to read from
        pdf_reader = PdfReader(pdf)
        #read the pages and add it to the text
        for page in pdf_reader.pages:
            #extracting the text from the pages and adding to the text variable
            text+=page.extract_text()
        return text #the output will be a string with allthe contents from the pdfs


#creating a function to convert our tetx to chunks
def get_text_chunks(raw_text):
    #in order to convert our text into chunks we are going to use a library called langchain (expecially a class from langchain called character text splitter)
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,# a "chunk" typically refers to a segment or portion of text that is processed as a unit. The concept of chunking is often used in tasks like tokenization, where text is divided into meaningful chunks for analysis.
        chunk_overlap=200,# chunk_overlap" typically refers to the extent of overlap between consecutive chunks or segments of text.(this is because sometine we need the next to overlap the previous chunk in order to have a proper menaing . if the chunk the starts where the previous chunk over then there will be trouble in understanding)
        length_function=len
    )

    chunks=text_splitter.split_text(raw_text)
    return chunks


#creating a function to convert the text  to chunks 
def get_vectorstore(text_chunks):
#---------------------------------------this is open api embedding code(priced) :no money no honey----------------------------------------------------------------------------------
   # embeddings = OpenAIEmbeddings()
#------------------------------------------------this one is a free embeding metho which need high processing power-----------------------------------------------------------------
    embeddings =HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks,embedding=embeddings)
    return vectorstore



#creating a conversation chain
def get_conversation_chain(vectorstore):
    llm = HuggingFaceHub(repo_id="lmsys/fastchat-t5-3b-v1.0", model_kwargs={"temperature":0.5, "max_length":512})
    memory = ConversationBufferMemory(memory_key='chat_history',return_messages=True)
    coneversation_chain = ConversationalRetrievalChain.from_llm(
            llm = llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
    )
    return coneversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question':user_question})
    st.session_state.chat_history = response['chat_history']

    for i,message in enumerate(st.session_state.chat_history):
        if i%2 == 0:
            st.write(user_template.replace("{{MSG}}",message.content),unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}",message.content),unsafe_allow_html=True)





def main():
    load_dotenv()


    st.set_page_config(page_title="Chat with multiple PDFs",page_icon=":books:")

    st.write(css,unsafe_allow_html=True)

    if "conversation" not in st.session_state :
        st.session_state.conversation = None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history= None
        
    st.header("Chat with multiple PDFs :books:")
    #creating a text box for user to ask questions:
    user_question = st.text_input("Ask a Question about your documents:")
    if user_question:
        handle_userinput(user_question)

   

    
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'",accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):

                # step1: get pdf text 
                raw_text = get_pdf_text(pdf_docs)
                


                #step 2: get the text chunks
                text_chunks=get_text_chunks(raw_text)
                #st.write(text_chunks)



                #step 3: create vector store (converting our chunks of text into vector embedings)
                vectorstore = get_vectorstore(text_chunks)
               

               # create conversation chain 
                st.session_state.conversation = get_conversation_chain(vectorstore)




if __name__=='__main__':
    main() 

