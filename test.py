from langchain_community.document_loaders import SeleniumURLLoader

from langchain_community.document_loaders import PlaywrightURLLoader
from langchain_community.document_loaders import WebBaseLoader
import bs4
import asyncio 
import sys

def web_search():
    urls = [
        "https://github.com/scrapy-plugins/scrapy-playwright/issues/7", "https://www.itf.gov.hk/"
    ]
    loader = WebBaseLoader(
        web_paths=(urls),
        bs_kwargs=dict(parse_only=bs4.SoupStrainer(name=["p", "div", "h[1-6]"])),
    )

    multiple_docs = loader.load()
    for i, docs in enumerate(multiple_docs):
        if docs.page_content == '': #dynamic web content
            loader = PlaywrightURLLoader(urls=[urls[i]], remove_selectors=["header", "footer"])
            multiple_docs[i] = loader.load()
            
    print(multiple_docs)

def web_search_test():
    urls = [
        "https://github.com/scrapy-plugins/scrapy-playwright/issues/7", "https://www.itf.gov.hk/"
    ]
    loader = PlaywrightURLLoader(urls=urls, remove_selectors=["header", "footer"])

    multiple_docs = loader.load()
            
    print(multiple_docs)

def web_search_selen():
    urls = [
        "https://github.com/scrapy-plugins/scrapy-playwright/issues/7", "https://www.itf.gov.hk/"
    ]
    loader = SeleniumURLLoader(urls=urls)

    multiple_docs = loader.load()
            
    print(multiple_docs)

a = [1,2]
a = del a[0]
print(a[0])


