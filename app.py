import validators,streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document
import re
import time
from deep_translator import GoogleTranslator
import textwrap
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "YT-Summarizer"
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"] 

groq_api_key= st.secrets["GROQ_API_KEY"] 



## sstreamlit APP
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

def extract_video_id(url):
    # Pattern for common YouTube video URLs
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    return None

## Get the Groq API Key and url(YT or website)to be summarized
with st.sidebar:
    groq_api_key=st.text_input("Groq API Key",value=groq_api_key,type="password")

generic_url=st.text_input("URL",label_visibility="collapsed")
print(generic_url)

## Gemma Model USsing Groq API
llm =ChatGroq(model="llama3-8b-8192", groq_api_key=groq_api_key)

prompt_template="""
Provide a summary of the following content in 500 words how it will impacted Indian stock market and which sector will be impacted and how:
Content:{text}

"""


max_retries = 10
retry_delay = 1  # seconds
chunk_size=2100
full_english_text =""

prompt=PromptTemplate(template=prompt_template,input_variables=["text"])

if st.button("Summarize the Content from YT or Website"):
    ## Validate all the inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid Url. It can may be a YT video utl or website url")

    else:
        try:
            with st.spinner("Waiting..."):

                if "youtube.com" in generic_url:

                    for attempt in range(max_retries):
                        try:

                            video_id = extract_video_id(generic_url)

                            lang_code=""

                            transcripts = YouTubeTranscriptApi.list_transcripts(video_id)

                            for cde in transcripts:

                                # print(f"Language: {cde.language}")
                                lang_code +=cde.language_code
                                print("---")
                                break
                            print(lang_code)
                            transcript = YouTubeTranscriptApi.get_transcript(video_id,languages=[lang_code])
                            full_hindi_text = " ".join([entry['text'] for entry in transcript])
                            # print(full_hindi_text)

                            hindi_wrap = textwrap.wrap(full_hindi_text, width=chunk_size, break_long_words=False, break_on_hyphens=False)
                            translator = GoogleTranslator(source=lang_code, target='en')

                            for hindi_words in hindi_wrap:
                                try:
                                    translated = translator.translate(hindi_words)
                                    full_english_text += translated

                                except Exception as e:
                                    print("Error translating chunk:", e)



                            print(full_english_text)
                            loader = [Document(page_content=full_english_text)]  # wrap in list directly
                            print(loader)

                            break



                        except Exception as e:
                            print(f"[Attempt {attempt + 1}] Error fetching transcript: {e}")
                            if attempt < max_retries - 1:
                                print(f"Retrying in {retry_delay} seconds...\n")
                                time.sleep(retry_delay)
                            else:
                                print("Max retries reached. Exiting.")

                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"}
                    )
                    loader = loader.load()

                docs = loader
                ## Chain For Summarization
                chain=load_summarize_chain(llm,chain_type="stuff",prompt=prompt)
                output_summary=chain.run(docs)

                st.success(output_summary)
        except Exception as e:
            st.exception(e)
