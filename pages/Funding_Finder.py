#For compatibility when deploying on streamlit
__import__('pysqlite3')
import sys
import sqlite3
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# print(sqlite3.sqlite_version_info)
import chromadb

import streamlit as st
from dotenv import load_dotenv

from ragllm import sequential_calls, results_processing, results_to_excel
from send_email import send_email
import requests
import nest_asyncio
import asyncio
import sys
# lower down security level
# requests.packages.urllib3.util.ssl_.DEFAULT_CIPHERS += ":HIGH:!DH:!aNULL"
requests.packages.urllib3.disable_warnings()

if "clicked" not in st.session_state:
    st.session_state.clicked = False


def click_button():
    st.session_state.clicked = True


def main():
    load_dotenv()
    type = "funding opportunity"
    st.set_page_config(
        page_title="Competition Finder",
        page_icon="👋",
    )

    st.write("# Welcome to Funding Finder! 👋")

    st.sidebar.success("Select the function above")
    keywords = st.text_input("What types of fundings are you finding?", key="keywords")
    email = st.text_input("Enter your email for receiving the results:", key="email")
    num = st.slider("Number of links to extract information for")
    st.button("Generate findings", on_click=click_button)

    if st.session_state.clicked:
        st.write("Generating results")
        results, url_list = sequential_calls(keywords, num, type)
        processed_results = results_processing(results, url_list)
        # Generate excel file
        excel_file_path = results_to_excel(processed_results, type)
        # send the file by email
        send_email(excel_file_path, email, type)
        st.write(f"Email sent to {email}!")
        #Clearing cache
        st.cache_resource.clear()


if __name__ == "__main__":
    requests.packages.urllib3.disable_warnings()
    # if sys.version_info >= (3, 8) and sys.platform.lower().startswith("win"):
    #     asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    # asyncio.run(main())
    main()
