# app.py

import streamlit as st
from pipeline import build_qa_chain

# Cache the chain to avoid reloading every time
@st.cache_resource(show_spinner="Building pipeline...")
def get_chain():
    return build_qa_chain()

def main():
    st.title("📄 PDF Q&A Assistant")

    role = st.text_input("Enter your Role", value="General")
    query = st.text_area("Ask your question", value="What are the titles of these pdf?")

    if st.button("Get Answer"):
        qa_chain = get_chain()
        result = qa_chain.invoke({"input": query, "role": role})
        sources = list(set(doc.metadata.get("source", "Unknown") for doc in result["context"]))

        st.subheader("🔍 Answer")
        st.write(result["answer"])

        st.subheader("📚 Sources")
        st.write(sources)

if __name__ == "__main__":
    main()
