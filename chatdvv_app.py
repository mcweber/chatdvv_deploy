# ---------------------------------------------------
# Version: 15.06.2024
# Author: M. Weber
# ---------------------------------------------------
# 05.06.2024 added searchFilter in st.session_state and sidebar
# 07.06.2024 implemented rag with fulltext search
# 09.06.2024 activated user management. Statistiken implemented.
# 15.06.2024 added filter for textsearch
# ---------------------------------------------------

import os
import streamlit as st
import pandas as pd
import chatdvv_module as myapi
import user_management

SEARCH_TYPES = {"vector": "Vector search", "llm": "LLM search", "rag": "RAG search", "fulltext": "Fulltext search"}

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
                if active_user != "":
                    st.session_state.userStatus = 'True'
                    st.session_state.userName = active_user
                    st.rerun()
                else:
                    st.error("User not found.")
            else:
                st.error("Please fill in all fields.")


@st.experimental_dialog("Statistiken")
def statistiken_dialog() -> None:
    st.write(f"Anzahl Artikel: {myapi.collection.count_documents({})}")
    st.write(f"Anzahl Artikel ohne Abstract: {myapi.collection.count_documents({'ki_abstract': ''})}")
    st.write(f"Anzahl Artikel ohne Embeddings: {myapi.collection.count_documents({'embeddings': {}})}")
#     st.write("-"*50)
#     st.write(myapi.group_by_field())
#     st.write(myapi.list_fields())
    if st.button("Close"):
        st.rerun()


# Main -----------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title='DVV Insight', initial_sidebar_state="collapsed")
    
    # Initialize Session State -----------------------------------------
    if 'userStatus' not in st.session_state:
        st.session_state.userStatus = False
        st.session_state.userName = ""
        st.session_state.searchStatus = False
        st.session_state.feldListe = list(myapi.group_by_field().keys())
        st.session_state.searchFilter = st.session_state.feldListe
        st.session_state.searchPref = "Artikel"
        st.session_state.searchResults = 5
        st.session_state.llmStatus = "openai"
        st.session_state.systemPrompt = "Du bist ein hilfreicher Assistent und gibst Informationen aus dem Bereich Transport und Verkehr."
        st.session_state.results = ""
        st.session_state.history = []
        st.session_state.searchType = "rag" # llm, vector, rag, fulltext
    if st.session_state.userStatus == False:
        login_user_dialog()
        # st.experimental_rerun()
    st.header("DVV Insight")
    # st.subheader("Research Bot für Fachartikel im Bereich Transport und Verkehr")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Version 0.1.4 - 15.06.2024")
    with col2:
        if st.session_state.userStatus:
            st.write(f"Eingeloggt als: {st.session_state.userName}")
        else:
            st.write("Nicht eingeloggt.")

    # Define Sidebar ---------------------------------------------------
    with st.sidebar:
        switch_searchFilter = st.multiselect(label="Choose Publications", options=st.session_state.feldListe, default=st.session_state.searchFilter)
        if switch_searchFilter != st.session_state.searchFilter:
            st.session_state.searchFilter = switch_searchFilter
            st.experimental_rerun()
        if st.button("Reset Filter"):
            st.session_state.searchFilter = st.session_state.feldListe
            st.experimental_rerun()
        # switch_searchType = st.radio(label="Choose Search Type", options=("rag", "llm", "vector", "fulltext"), index=0)
        # if switch_searchType != st.session_state.searchType:
        #     st.session_state.searchType = switch_searchType
        #     st.experimental_rerun()
        switch_search_results = st.slider("Search Results", 1, 20, st.session_state.searchResults)
        if switch_search_results != st.session_state.searchResults:
            st.session_state.searchResults = switch_search_results
            st.experimental_rerun()
        switch_llm = st.radio(label="Switch to LLM", options=("groq", "openai"), index=1)
        if switch_llm != st.session_state.llmStatus:
            st.session_state.llmStatus = switch_llm
            st.experimental_rerun()
        switch_SystemPrompt = st.text_area("System-Prompt", st.session_state.systemPrompt)
        if switch_SystemPrompt != st.session_state.systemPrompt:
            st.session_state.systemPrompt = switch_SystemPrompt
            st.experimental_rerun()
        if st.button("Statistiken"):
            statistiken_dialog()
        if st.button("Logout"):
            st.session_state.userStatus = False
            st.session_state.searchStatus = False
            st.session_state.userName = ""
            st.experimental_rerun()

    # Define Search Type ------------------------------------------------
    switch_searchType = st.radio(label="Choose Search Type", options=("rag", "llm", "vector", "fulltext"), index=0, horizontal=True)
    if switch_searchType != st.session_state.searchType:
        st.session_state.searchType = switch_searchType
        st.experimental_rerun()

    # Define Search Form ----------------------------------------------
    with st.form(key="searchForm"):
        question = st.text_input(SEARCH_TYPES[st.session_state.searchType])
        if st.session_state.searchType in ["rag", "llm"]:
            button_caption = "Fragen"
        else:
            button_caption = "Suchen"
        if st.form_submit_button(button_caption) and question != "":
            st.session_state.searchStatus = True
        
    # Define Search & Search Results -------------------------------------------
    if st.session_state.userStatus and st.session_state.searchStatus:
        if st.session_state.searchType == "fulltext":
            results, results_count = myapi.text_search(
                search_text=question, 
                filter=st.session_state.searchFilter, 
                limit=st.session_state.searchResults
                )
            # st.caption(f'Suche nach: "{question}". {results_count} Artikel gefunden.')
            df = pd.DataFrame(results)
            output = df[["quelle_id", "nummer", "jahrgang", "titel", "text"]]
            st.dataframe(output)
            # counter = 1
            # for result in results:
            #     # st.write(f"[{result['datum']}] {result['titel']}")
            #     st.write(f"[{result['quelle_id']}, {result['nummer']}/{result['jahrgang']}] {result['titel']}")
            #     st.write(result['text'][:500] + " ...")
            #     st.divider()
            #     counter += 1
            #     if counter > st.session_state.searchResults:
            #         break
        elif st.session_state.searchType == "vector":
            results = myapi.vector_search(
                query_string=question, 
                limit=st.session_state.searchResults
                )
            for result in results:
                # st.write(f"[{result['datum']}] {result['titel']}")
                st.write(f"[{result['quelle_id']}, {result['nummer']}/{result['jahrgang']}] {result['titel']}")
                st.write(result['text'])
                st.divider()
        elif st.session_state.searchType == "llm":
            summary = myapi.ask_llm(
                llm=st.session_state.llmStatus,
                temperature=0.2,
                question=question,
                history=[],
                systemPrompt=st.session_state.systemPrompt,
                results_str=""
                )
            st.write(summary)
        elif st.session_state.searchType == "rag":
            # results = myapi.vector_search(question, st.session_state.searchResults)
            results, results_count = myapi.text_search(
                search_text=question, 
                filter=st.session_state.searchFilter, 
                limit=st.session_state.searchResults
                )
            with st.expander("DB Suchergebnisse"):
                results_str = ""
                for result in results:
                    st.write(f"[{result['quelle_id']}, {result['nummer']}/{result['jahrgang']}] {result['titel']}")
                    results_str += f"Datum: {result['datum']}\nTitel: {result['titel']}\nText: {result['text']}\n\n"
            summary = myapi.ask_llm(
                llm=st.session_state.llmStatus,
                temperature=0.2,
                question=question,
                history=[],
                systemPrompt=st.session_state.systemPrompt,
                results_str=results_str
                )
            st.write(summary)
        st.session_state.searchStatus = False


if __name__ == "__main__":
    main()
