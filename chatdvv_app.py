# ---------------------------------------------------
# Version: 13.07.2024
# Author: M. Weber
# ---------------------------------------------------
# 05.06.2024 added searchFilter in st.session_state and sidebar
# 07.06.2024 implemented rag with fulltext search
# 09.06.2024 activated user management. Statistiken implemented.
# 15.06.2024 added filter for textsearch
# 16.06.2024 switched rag to vector search
# 16.06.2024 added Markbereiche as search filter
# 16.06.2023 added user role. sidebar only for admin
# 21.06.2024 added switch between fulltext and vector search for rag
# 22.06.2024 added anthropic as llm
# 22.06.2024 added web search for rag
# 30.06.2024 switched websearch to news-search
# 06.07.2024 added document view
# 06.07.2024 added tavily web search
# 07.07.2024 added document info window
# 08.07.2024 added score in search results
# 10.07.2024 switched input to st.caption, moved product selection to bottom of sidebar
# 12.07.2024 added INDUSTR contents
# 12.07.2024 added show latest articles
# 13.07.2024 added industry filter to vector search
# 13.07.2024 added number of documents per publication to statistiken
# ---------------------------------------------------

import streamlit as st
import chatdvv_module as myapi
import user_management

SEARCH_TYPES = ("rag", "llm", "vektor", "volltext", "web")
MARKTBEREICHE = ("Alle", "Logistik", "Maritim", "Rail", "ÖPNV", "Industrie")
PUB_LOG = ("THB", "DVZ", "DVZT", "THBT", "DVZMG", "DVZM", "DVZ-Brief")
PUB_MAR = ("THB", "THBT", "SHF", "SHIOF", "SPI", "NSH")
PUB_RAIL =("EI", "SD", "BM", "BAMA")
PUB_OEPNV = ("RABUS", "NAHV", "NANA", "DNV")
PUB_PI = ("pi_AuD", "pi_PuA", "pi_EuE", "pi_E20", "pi_Industry_Forward", "pi_Industrial_Solutions", "pi_Next_Technology", "pi_")
MARKTBEREICHE_LISTE = (PUB_LOG, PUB_MAR, PUB_RAIL, PUB_OEPNV, PUB_PI)

# Functions -------------------------------------------------------------

@st.experimental_dialog("Login User")
def login_user_dialog() -> None:
    with st.form(key="loginForm"):
        st.write(f"Status: {st.session_state.userStatus}")
        user_name = st.text_input("Benutzer")
        user_pw = st.text_input("Passwort", type="password")
        if st.form_submit_button("Login"):
            if user_name and user_pw:
                active_user = user_management.check_user(user_name, user_pw)
                if active_user:
                    st.session_state.userName = active_user["username"]
                    st.session_state.userRole = active_user["rolle"]
                    st.session_state.userStatus = 'True'
                    st.rerun()
                else:
                    st.error("User not found.")
            else:
                st.error("Please fill in all fields.")

@st.experimental_dialog("Statistiken")
def statistiken_dialog() -> None:
    num_documents = myapi.collection.count_documents({})
    num_abstracts = myapi.collection.count_documents({'ki_abstract': {'$ne': ''}})
    num_embeddings = myapi.collection.count_documents({'embeddings': {'$ne': {}}})
    st.write(f"Anzahl Artikel: {num_documents:,}".replace(",", "."))
    st.write(f"Anzahl Artikel mit Abstracts: {num_abstracts:,}".replace(",", "."))
    st.write(f"Anzahl Artikel ohne Embeddings: {num_documents - num_embeddings:,}".replace(",", "."))
    st.divider()
    st.write("Anzahl Artikel pro Marktbereich:")
    for item in MARKTBEREICHE_LISTE:
        st.write(f"{item}: {myapi.collection.count_documents({'quelle_id': {'$in': item}}):,}".replace(",", "."))

@st.experimental_dialog("DokumentenAnsicht")
def document_view(result: list = "Kein Text übergeben.") -> None:
    st.title(result['titel'])
    st.write(f"[{round(result['score'], 3)}] {result['quelle_id']}, {result['nummer']}/{result['jahrgang']} vom {str(result['date'])[:10]}\n\n[Score: {result['score']}]")
    st.write(result['text'])
    st.session_state.searchStatus = True

@st.experimental_dialog("DokumentenInfo")
def document_info(result: list = "Kein Text übergeben.") -> None:
    st.header(result['titel'])
    st.write(f"{result['quelle_id']}, {result['nummer']}/{result['jahrgang']} vom {str(result['date'])[:10]}")
    st.subheader("Zusammenfassung:")
    st.write(myapi.write_summary(text=result['text'], length=100))
    st.subheader("Takeaways:")
    st.write(myapi.write_takeaways(text=result['text']))
    st.subheader("Schlagworte:")
    st.write(myapi.generate_keywords(text=result['text'], max_keywords=5))
    st.session_state.searchStatus = True

def print_results(results: list, max_items: int = 100) -> None:
    counter = 1
    for result in results:
        col = st.columns([0.8, 0.1, 0.1])
        with col[0]:
            st.write(f"[{round(result['score'], 3)}][{result['quelle_id']}, {result['nummer']}/{result['jahrgang']}] {result['titel']}")
        with col[1]:
            st.button(label="DOC", key=str(result['_id'])+"DOC", on_click=document_view, args=(result,))
        with col[2]:
            st.button(label="INFO", key=str(result['_id'])+"INFO", on_click=document_info, args=(result,))
        counter += 1
        if counter > max_items:
            break

def show_latest_articles(max_items: int = 10) -> None:
    st.write("Neueste Artikel")
    results = myapi.text_search(
        search_text="*",
        score=0.0,
        filter=st.session_state.searchFilter,
        limit=max_items
        )
    print_results(results)

# Main -----------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title='DVV Insight', initial_sidebar_state="collapsed")
    
    # Initialize Session State -----------------------------------------
    if 'userStatus' not in st.session_state:
        st.session_state.feldListe: list = list(myapi.group_by_field().keys())
        st.session_state.history: list = []
        st.session_state.llmStatus: str = myapi.LLMS[0]
        st.session_state.marktbereich: str = "Alle"
        st.session_state.marktbereichIndex: int = 0
        st.session_state.rag_db_suche: bool = True
        st.session_state.rag_web_suche: bool = True
        st.session_state.rag_index: str = "vektor" # fulltext, vektor
        st.session_state.results: str = ""
        st.session_state.searchFilter: list = st.session_state.feldListe
        st.session_state.searchPref: str = "Artikel"
        st.session_state.searchResultsLimit:int  = 50
        st.session_state.searchStatus: bool = False
        st.session_state.searchType: str = "rag"
        st.session_state.searchTypeIndex: int  = SEARCH_TYPES.index(st.session_state.searchType)
        st.session_state.showLatest: bool = False
        st.session_state.systemPrompt: str = myapi.get_systemprompt()
        st.session_state.userName: str = ""
        st.session_state.userRole: str = ""
        st.session_state.userStatus: bool = False
        st.session_state.webSearch: str = "tavily" # tavily, DDGS
   
    if st.session_state.userStatus == False:
        login_user_dialog()
    st.header("DVV Insight")
    col = st.columns(2)
    with col[0]:
        st.caption("Version: 13.07.2024 Status: POC")
    with col[1]:
        if st.session_state.userStatus:
            st.caption(f"Eingeloggt als: {st.session_state.userName}")
        else:
            st.caption("Nicht eingeloggt.")

    # Define Sidebar ---------------------------------------------------
    if st.session_state.userRole == "admin":
        with st.sidebar:
            if st.button("Statistiken"):
                statistiken_dialog()
            st.divider()
            switch_search_results = st.slider("Search Results", 1, 100, st.session_state.searchResultsLimit)
            if switch_search_results != st.session_state.searchResultsLimit:
                st.session_state.searchResultsLimit = switch_search_results
                st.experimental_rerun()
            switch_rag_db_suche = st.checkbox("DB-Suche", value=st.session_state.rag_db_suche)
            if switch_rag_db_suche != st.session_state.rag_db_suche:
                st.session_state.rag_db_suche = switch_rag_db_suche
                st.experimental_rerun()
            switch_rag_web_suche = st.checkbox("WEB-Suche", value=st.session_state.rag_web_suche)
            if switch_rag_web_suche != st.session_state.rag_web_suche:
                st.session_state.rag_web_suche = switch_rag_web_suche
                st.experimental_rerun()
            switch_llm = st.radio(label="Switch LLM", options=myapi.LLMS, index=0)
            if switch_llm != st.session_state.llmStatus:
                st.session_state.llmStatus = switch_llm
                st.experimental_rerun()
            switch_rag_index = st.radio(label="Switch RAG-index", options=("fulltext", "vektor"), index=0)
            if switch_rag_index != st.session_state.rag_index:
                st.session_state.rag_index = switch_rag_index
                st.experimental_rerun()
            switch_webSearch = st.radio(label="Switch Web-Suche", options=("tavily", "DDGS"), index=0)
            if switch_webSearch != st.session_state.webSearch:
                st.session_state.webSearch = switch_webSearch
                st.experimental_rerun()
            st.divider()
            switch_SystemPrompt = st.text_area("System-Prompt", st.session_state.systemPrompt, height=500)
            if switch_SystemPrompt != st.session_state.systemPrompt:
                st.session_state.systemPrompt = switch_SystemPrompt
                myapi.update_systemprompt(switch_SystemPrompt)
                st.experimental_rerun()
            st.divider()
            switch_searchFilter = st.multiselect(label="Choose Publications", options=st.session_state.feldListe, default=st.session_state.searchFilter)
            if switch_searchFilter != st.session_state.searchFilter:
                st.session_state.searchFilter = switch_searchFilter
                st.experimental_rerun()
            if st.button("Reset Filter"):
                st.session_state.searchFilter = st.session_state.feldListe
                st.session_state.marktbereich = "Alle"
                st.session_state.marktbereichIndex = 0
                st.experimental_rerun()
            st.divider()
            if st.button("Logout"):
                st.session_state.userStatus = False
                st.session_state.searchStatus = False
                st.session_state.userName = ""
                st.experimental_rerun()

    # Define Search Type ------------------------------------------------
    switch_searchType = st.radio(label="Auswahl Suchtyp", options=SEARCH_TYPES, index=st.session_state.searchTypeIndex, horizontal=True)
    if switch_searchType != st.session_state.searchType:
        st.session_state.searchType = switch_searchType
        st.session_state.searchTypeIndex = SEARCH_TYPES.index(switch_searchType)
        st.experimental_rerun()
    
    # Define Search Filter ----------------------------------------------
    switch_marktbereich = st.radio(label="Auswahl Marktbereich", options=MARKTBEREICHE, index=st.session_state.marktbereichIndex, horizontal=True)
    if switch_marktbereich != st.session_state.marktbereich:
        if switch_marktbereich == "Logistik":
            st.session_state.searchFilter = PUB_LOG
        elif switch_marktbereich == "Maritim":
            st.session_state.searchFilter = PUB_MAR
        elif switch_marktbereich == "Rail":
            st.session_state.searchFilter = PUB_RAIL
        elif switch_marktbereich == "ÖPNV":
            st.session_state.searchFilter = PUB_OEPNV
        elif switch_marktbereich == "Industrie":
            st.session_state.searchFilter = PUB_PI
        elif switch_marktbereich == "Alle":
            st.session_state.searchFilter = st.session_state.feldListe
        st.session_state.marktbereich = switch_marktbereich
        st.session_state.marktbereichIndex = MARKTBEREICHE.index(switch_marktbereich)
        st.experimental_rerun()

    # Define Search Form ----------------------------------------------
    with st.form(key="searchForm"):
        question = st.text_area(f"{st.session_state.searchType} [{st.session_state.rag_index}]")
        if st.session_state.searchType in ["rag", "llm"]:
            button_caption = "Fragen"
        else:
            button_caption = "Suchen"
        col = st.columns([0.7, 0.3])
        with col[0]:
            if st.form_submit_button(button_caption) and question != "":
                st.session_state.searchStatus = True
        with col[1]:
            if st.session_state.searchType == "volltext" and st.form_submit_button("Neueste Artikel"):
                st.session_state.showLatest = True

    # Show Latest Articles ---------------------------------------------
    if st.session_state.searchType == "volltext" and st.session_state.showLatest:
        show_latest_articles(max_items=st.session_state.searchResultsLimit)
        st.session_state.showLatest = False

    # Define Search & Search Results -------------------------------------------
    if st.session_state.userStatus and st.session_state.searchStatus:
        # Fulltext Search -------------------------------------------------
        if st.session_state.searchType == "volltext":
            results = myapi.text_search(
                search_text=question,
                score=5.0,
                filter=st.session_state.searchFilter, 
                limit=st.session_state.searchResultsLimit
                )
            print_results(results, st.session_state.searchResultsLimit)
        # Vector Search --------------------------------------------------        
        elif st.session_state.searchType == "vektor":
            results = myapi.vector_search(
                query_string=question,
                score=0.5,
                filter=st.session_state.searchFilter,
                limit=st.session_state.searchResultsLimit
                )
            print_results(results, st.session_state.searchResultsLimit)
        # LLM Search -----------------------------------------------------
        elif st.session_state.searchType == "llm":
            summary = myapi.ask_llm(
                llm=st.session_state.llmStatus,
                temperature=0.2,
                question=question,
                history=[],
                systemPrompt=st.session_state.systemPrompt,
                results_str="",
                web_results_str=""
                )
            st.write(summary)
        # WEB Search -----------------------------------------------------
        elif st.session_state.searchType == "web":
            if st.session_state.webSearch == "tavily":
                results = myapi.web_search_tavily(query=question, score=0.5, limit=10)
            else:
                results = myapi.web_search_ddgs(query=question, limit=10)
            if results:
                for result in results:
                    if st.session_state.webSearch == "tavily":
                        st.write(f"[{round(result['score'], 3)}] {result['title'][:10]} [{result['url']}]")
                    else:
                        st.write(f"{result['title']} [{result['href']}]")
            else:
                st.write("WEB-Suche bringt keine Ergebnisse.")
        # RAG Search -----------------------------------------------------
        elif st.session_state.searchType == "rag":
            # DB Search -------------------------------------------------
            db_results_str = ""
            if st.session_state.rag_db_suche:
                if st.session_state.rag_index == "vektor":
                    results = myapi.vector_search(
                    query_string=question,
                    score=0.5,
                    filter=st.session_state.searchFilter, 
                    limit=st.session_state.searchResultsLimit
                    )
                else:
                    results = myapi.text_search(
                        search_text=question, 
                        score=5.0,
                        filter=st.session_state.searchFilter, 
                        limit=st.session_state.searchResultsLimit
                        )
                with st.expander("DVV-Archiv Suchergebnisse"):
                    for result in results:
                        col = st.columns([0.7, 0.1, 0.2])
                        with col[0]:
                            st.write(f"[{round(result['score'], 3)}][{result['quelle_id']}, {result['nummer']}/{result['jahrgang']}] {result['titel']}")
                        with col[1]:
                            st.button(label="DOC", key=str(result['_id'])+"DOC", on_click=document_view, args=(result,))
                        with col[2]:
                            st.button(label="INFO", key=str(result['_id'])+"INFO", on_click=document_info, args=(result,))
                        db_results_str += f"Datum: {str(result['date'])[:10]}\nTitel: {result['titel']}\nText: {result['text']}\n\n"
            else:
                st.write("DVV-Archiv-Suche bringt keine Ergebnisse.")
            # Web Search ------------------------------------------------
            web_results_str = ""
            if st.session_state.rag_web_suche:
                if st.session_state.webSearch == "tavily":
                    results = myapi.web_search_tavily(query=question, score=0.5, limit=10)
                    with st.expander("WEB Suchergebnisse"):
                        for result in results:
                            st.write(f"[{round(result['score'], 3)}] {result['title']} [{result['url']}]")
                            web_results_str += f"Titel: {result['title']}\nURL: {result['url']}\nText: {result['raw_content']}\n\n"
                else:
                    results = myapi.web_search_ddgs(query=question, limit=10)
                    with st.expander("WEB Suchergebnisse"):
                        for result in results:
                            st.write(f"{result['title']} [{result['href']}]")
                            web_results_str += f"Titel: {result['title']}\nURL: {result['href']}\nText: {result['body']}\n\n"
            # LLM Search ------------------------------------------------
            summary = myapi.ask_llm(
                llm=st.session_state.llmStatus,
                temperature=0.2,
                question=question,
                history=[],
                systemPrompt=st.session_state.systemPrompt,
                db_results_str=db_results_str,
                web_results_str=web_results_str
                )
            st.write(summary)
        st.session_state.searchStatus = False

if __name__ == "__main__":
    main()
