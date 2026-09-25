"""Streamlit chat UI for the HR Policy Assistant (Cloud Run service: hr-rag-assistant)."""

import logging

import streamlit as st

from hr_rag.auth import is_email_allowed
from hr_rag.config import get_settings
from hr_rag.factory import build_assistant

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

st.set_page_config(page_title="HR Policy Assistant", page_icon="📘", layout="centered")
settings = get_settings()


# --- Layer 1: Google OAuth login + employee allow-list -------------------------------------
def current_user_email() -> str:
    if settings.auth_disabled:
        st.sidebar.warning("Auth disabled (local development mode)")
        return "dev@localhost"

    if not st.user.is_logged_in:
        st.title("📘 HR Policy Assistant")
        st.write("Sign in with your company Google account to ask questions about HR policies.")
        st.button("Sign in with Google", type="primary", on_click=st.login)
        st.stop()

    email = st.user.get("email")
    if not st.user.get("email_verified", False) or not is_email_allowed(
        email, settings.allowed_emails, settings.allowed_domains
    ):
        st.error(f"{email} is not authorised to use this assistant. Contact HR if this is a mistake.")
        st.button("Sign out", on_click=st.logout)
        st.stop()
    return email


email = current_user_email()


@st.cache_resource(show_spinner="Starting assistant...")
def get_assistant():
    return build_assistant(settings)


# --- Sidebar --------------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"Signed in as **{email}**")
    if not settings.auth_disabled:
        st.button("Sign out", on_click=st.logout)
    if st.button("New conversation"):
        st.session_state.messages = []
        st.rerun()
    st.caption(
        "Answers come only from official HR policy documents and include citations. "
        "For personal cases, contact HR directly."
    )


# --- Chat -----------------------------------------------------------------------------------
st.title("📘 HR Policy Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_citations(citations: list[dict]) -> None:
    if not citations:
        return
    with st.expander(f"Sources ({len(citations)})"):
        for c in citations:
            page = f", p. {c['page']}" if c.get("page") else ""
            st.markdown(f"**[{c['number']}] {c['title']}**{page}  \n`{c['source']}`")
            st.caption(c["excerpt"] + ("…" if len(c["excerpt"]) >= 500 else ""))


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_citations(message.get("citations", []))
        if message.get("meta"):
            st.caption(message["meta"])

if prompt := st.chat_input("Ask about leave, benefits, remote work, expenses..."):
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching HR policies..."):
            try:
                response = get_assistant().answer(prompt, history=history, user_email=email)
            except Exception:
                logging.exception("Assistant failed")
                st.error("Something went wrong while answering. Please try again in a moment.")
                st.stop()

        citations = [
            {"number": c.number, "title": c.title, "source": c.source, "page": c.page, "excerpt": c.excerpt}
            for c in response.citations
        ]
        meta_parts = []
        if response.cache_hit:
            meta_parts.append("⚡ cached answer")
        if response.model:
            meta_parts.append(response.model)
        meta_parts.append(f"{response.latency_ms / 1000:.1f}s")
        meta = " · ".join(meta_parts)

        st.markdown(response.answer)
        render_citations(citations)
        st.caption(meta)

    st.session_state.messages.append(
        {"role": "assistant", "content": response.answer, "citations": citations, "meta": meta}
    )
