import streamlit as st
import pandas as pd
from io import BytesIO
from tavily import TavilyClient
from langchain_openai import AzureChatOpenAI
import json
import requests
import os
from dotenv import load_dotenv
import ast
import time
import re
import os
import threading
import functools

load_dotenv()

write_lock = threading.Lock()
csv_buffer = BytesIO()
data=csv_buffer
MAX_RETRIES = 1
RETRY_DELAY = 1  # 1 seconds on 429 error


tavily_client = TavilyClient(api_key=os.getenv('TAVILY_API_KEY'))

# initialize llm
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
    azure_deployment=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
    openai_api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    openai_api_version="2024-05-01-preview",
    openai_api_type="azure"
)
@functools.lru_cache(maxsize=128)
def cached_extraction(manufacturer: str, part_number: str) -> str:
    tool_input = f"{manufacturer}||{part_number}"
    return extraction(tool_input)

@functools.lru_cache(maxsize=128)
def cached_google_search(manufacturer: str, part_number: str) -> str:
    tool_input = f"{manufacturer}||{part_number}"
    return google_search(tool_input)

def highlight_alternate_rows(row):
    return ['background-color: #ccff99' if row.name % 2 else 'background-color: #ffe6ff'] * len(row)

# tavily search function
def tavily_search(query: str) -> dict:
    res = tavily_client.search(
        query=query,
        search_depth="advanced",
        include_answer="advanced",
        include_raw_content=True,
        max_results=4
    )
    results = {
    "Answer": res['answer']}
    return results


# Google Search
def google_search(query: str) -> str:
    parts = query.split("||")
    if len(parts) == 2:
            manufacturer = parts[0].strip()
            part_number = parts[1].strip()
            print(part_number)
    res = requests.get(
        "https://google.serper.dev/search",
        params={"q": f"Find the official datasheet URL for part number {part_number} from {manufacturer}. The URL should point to a genuine and authoritative source such as the manufacturer's website or a trusted distributor"},
        headers={"X-API-KEY": os.getenv('SERPAPI_KEY')}
    )
    print("+++++++++++++++++++++++++")
    if res.json()['organic'][0]['link']:
        print(str(res.json()['organic'][0]['link']))
    print("+++++++++++++++++++++++++")
    resultURL = llm.invoke('''Give URL of the Datasheet for Part Number('''+str({part_number})+''' of Manufacturer '''+str({manufacturer})+'''. Return in JSON format with 'Datasheet_URL'as key, value of 'Datasheet_URL' should be a link only else value is set to 'Not Found'. Strictly follow this format
                                {
                                    "Datasheet_URL":""
                                }
                            Query: ''')+str(res.json()['organic'][0])
    # print(resultURL)
    last_msg = resultURL.messages[-1]
    template_str = last_msg.prompt.template
    data = ast.literal_eval(template_str)
    link_url = data["link"]
    print(link_url)
    return link_url
# Functions
def extraction(input_str: str) -> str:
    try:
        parts = input_str.split("||")
        if len(parts) == 2:
            manufacturer = parts[0].strip()
            part_number = parts[1].strip()
    except ValueError:
        return "Error: Input must be in the format 'Manufacturer||PartNumber||2'."
    query={"promptSES":"","promptURL":""}
    ans1= tavily_search(f"Check if part number {part_number} from {manufacturer} is currently available or discontinued. If discontinued, provide the end-of-life (EOL) date and the replacement or superseding part number, if any. Ensure the information is sourced from a genuine and authoritative source such as the manufacturer’s website or a trusted distributor.")
    # ans1= tavily_search(f"Is the part number {part_number} of {manufacturer} is available or discontinued? If discontinued, what is the date for its end of life (follow format DD-MM-YYYY) and superseded by which Part Number component? Also provide the URL or name of website from where the response can be verified.")
    print("Tavily Response for Status EOL and supersed by")
    print(ans1)
    query=ans1
    return query

def extract_json_from_response(response_text: str) -> dict:
        matchi = re.search(r'\{.*?\}', response_text, re.DOTALL)
        pattern = r'\{[^{}]*"Status"[^{}]*\}'

        match = re.search(pattern, matchi.group())
        print("JSON FORMAT : ")
        
        if match:
            print(match.group())
            json_str = match.group()
            json_str = re.sub(r'\bNone\b', 'null', json_str)
            json_str = json_str.replace('"None"', 'null')
            print("JSON FORMAT : ")
            print(json_str)

            try:
                print(json.loads(json_str))
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print("Error decoding JSON:", e)
                return None
        else:
            print("No JSON found.")
        return None
        
def call_agent_with_retry(prompt1, retries=MAX_RETRIES):
            for attempt in range(retries):
                try:
                    resultSES = llm.invoke(
                        '''Give Status(available/Discontinued) by default Status will be set to available, if status is discontinued then what is End_Of_Life_Date in 'DD-MMM-YYYY' format for example 09-May-2020 (by default End_Of_Life_Date will be set to None) else if available then set End_Of_Life_Date to None and which Part Number Superseded it (by default Superseded_By will be set to None).Extract Verified from which website. Strictly follow this format 
                                {
                                "Status":"",
                                "End_Of_Life_Date":"",
                                "Superseded_By":"",
                                "Verified_From":""
                                }.
                                Query: '''+str(prompt1['Answer'])
                    )
                    dataSES = extract_json_from_response(resultSES.content) 
                    print("Extracted Status, EOL and Superseded_By")
                    print(dataSES)
                    if dataSES is not None:
                        dataSES['raw_content'] = str(resultSES.content)
                        return dataSES
                    else:
                        dataSES = {}
                        return dataSES
                except Exception as e:
                        print(e)
                        if "429" in str(e):
                            wait_time = RETRY_DELAY * (attempt + 1)
                            print(f"[⚠️] Rate limit hit. Retrying in {wait_time} seconds (Attempt {attempt + 1}/{retries})...")
                            time.sleep(wait_time)
                        else:
                            raise e
                raise RuntimeError("❌ Exceeded maximum retry attempts due to rate limits.")

def process_async_google(idx,row,df):
    manufacturer = str(row["Manufacturer_Name"]).strip()
    part_number = str(row["Part_Number"]).strip()
    tool_input = f"{manufacturer}||{part_number}"
    prompt = google_search(tool_input)
    print("Google Search : ")
    print(prompt)
    write_lock.acquire()
    if prompt is not None:
        try:
                        if prompt and prompt not in ['"null"', None,'None','null'] :
                            df.at[idx, "Datasheet_URL"] = str(prompt) 
                        else:
                            df.at[idx, "Datasheet_URL"] = "Not Found"
        except Exception as e:
                        print(f"Exception occured for Datasheet_URL : {e}")
    else: 
           print("No Response from LLM.")                

    print(f"[✅] Processed row {idx} ({manufacturer}, {part_number})")
    write_lock.release()
def process_async(idx,row,df):
    manufacturer = str(row["Manufacturer_Name"]).strip()
    part_number = str(row["Part_Number"]).strip()
    tool_input = f"{manufacturer}||{part_number}"
    # prompt = extraction(tool_input)
    # link = google_search(tool_input)
    prompt = cached_extraction(manufacturer, part_number)
    link = cached_google_search(manufacturer, part_number)
    result = None
    try:
                try:
                    result = call_agent_with_retry(prompt)
                    print("My Info : -----------------------")
                    print(result)
                    
                except json.JSONDecodeError:
                    print(f"[⚠️] JSON parsing failed for row {idx}. Skipping...")
    except Exception as e:
                print(f"[❌] Error processing row {idx}: {e}")
    write_lock.acquire()
    if result is not None:
        try:
                        if result.get('Status') :
                            df.at[idx, "Status"] = str(result['Status']) 
                        else:
                            df.at[idx, "Status"] = "Not Found"
        except Exception as e:
                        print(f"Exception occurred for Status: {e}")
                    
        try:
                        if result.get('End_Of_Life_Date') and result.get('End_Of_Life_Date') not in ['"null"', None,'None','null']:
                            df.at[idx, "End_Of_Life_Date"] = str(result['End_Of_Life_Date']) 
                        else:
                            df.at[idx, "End_Of_Life_Date"] = "Not Found"
                        
        except Exception as e:
                        print(f"Exception occurred for End_Of_Life_Date: {e}")
                    
        try:
                        if result.get('Superseded_By') and result.get('Superseded_By') not in ['"null"', None,'None','null']:
                            df.at[idx, "Superseded_By"] = str(result['Superseded_By']) 
                        else:
                            df.at[idx, "Superseded_By"] = "Not Found"
                        
        except Exception as e:
                        print(f"Exception occured for Superseded_By : {e}")
        
        try:
                        if result.get('Verified_From') and result.get('Verified_From') not in ['"null"', None,'None','null']:
                            df.at[idx, "Verified_From(Status,EOL,SB)"] = str(result['Verified_From']) 
                        else:
                            df.at[idx, "Verified_From(Status,EOL,SB)"] = "Not Found"
                        
        except Exception as e:
                        print(f"Exception occured for Verified_From(Status,EOL,SB) : {e}")
                        
        try:
                        if result.get('raw_content'):
                            df.at[idx, "raw_content"] = str(result['raw_content'])+"|| Datasheet: "+str(link)  
                        else:
                            df.at[idx, "raw_content"] = "Not Found"
                        
        except Exception as e:
                        print(f"Exception occured for raw_content : {e}")
                        
        try:
                        if link and link not in ['"null"', None,'None','null'] :
                            df.at[idx, "Datasheet_URL"] = str(link) 
                        else:
                            df.at[idx, "Datasheet_URL"] = "Not Found"
        except Exception as e:
                        print(f"Exception occured for Datasheet_URL : {e}")
    else: 
           print("No Response from LLM.")                

    print(f"[✅] Processed row {idx} ({manufacturer}, {part_number})")
    write_lock.release()

# Streamlit UI
st.set_page_config(page_title="Component Availability Checker", page_icon="📦")
st.markdown(
    """
    <style>
        .stApp {
            background-color: #8FBC8F;
        }
    </style>
    """,
    unsafe_allow_html=True
)
st.title("📦 Component Availability Tracker")

st.markdown("Upload a CSV file with **Manufacturer_Name** and **Part_Number** columns.")
col1, col2 = st.columns([1,4])
with col1:
    batch_size_input = st.text_input("**Batch Size(default 2)**",value="2",)
    
with col2:
    uploaded_file = st.file_uploader("**Upload a CSV file**", type=["csv"])
if uploaded_file:
    batch_size = int(batch_size_input)
    df = pd.read_csv(uploaded_file)  

    # Ensure columns exist
    if "Manufacturer_Name" not in df.columns or "Part_Number" not in df.columns:
        st.error("CSV must contain 'Manufacturer_Name' and 'Part_Number' columns.")
        st.stop()
    else:
        for col in ["Status", "End_Of_Life_Date","Superseded_By", "Verified_From(Status,EOL,SB)","Datasheet_URL", "raw_content"]:
            if col not in df.columns:
                df[col] = ""
        
        df["Status"] = df["Status"].astype(str)
        df["End_Of_Life_Date"] = df["End_Of_Life_Date"].astype(str)
        df["Superseded_By"] = df["Superseded_By"].astype(str)
        df["Verified_From(Status,EOL,SB)"] = df["Verified_From(Status,EOL,SB)"].astype(str)
        df["Datasheet_URL"] = df["Datasheet_URL"].astype(str)
        df["raw_content"] = df["raw_content"].astype(str)

        
        # Progress UI
        progress_bar = st.progress(0)
        status_text = st.empty()
        preview_holder = st.empty()
        threads = []
        manu = []
        # batch_size = int(os.getenv(batch_size_input, 2))
        for i, (idx,row) in enumerate(df.iterrows()):
            # async part
                t = threading.Thread(target=process_async, args=(idx,row,df))
                t.start()
                manu.append(str(row['Manufacturer_Name']).strip())
                threads.append(t)
                if len(threads) == batch_size or i == len(df) - 1:
                    status_text.markdown("Processing: "+str(manu) +"[ "+str(percent_complete)+" %]")
                     
                    # Wait for all threads to complete
                    for t in threads:
                        t.join()
                    threads.clear()
                    # for n in manu:
                    #        st.markdown(n)
                    #        st.dataframe(df[df['Manufacturer_Name'] == n])
                    #     #    print(n)
                    manu.clear()
                    print("batch over")
                
                percent_complete = int(((i) / len(df)) * 100)
                progress_bar.progress((i) / len(df))
                
    uploaded_file.name = 'xyz'
    progress_bar.progress(100)
    percent_complete = int(100)
    status_text.markdown(f"Processed :({percent_complete} %)")
    styled_df = df.style.apply(highlight_alternate_rows, axis=1)
    st.dataframe(styled_df)

