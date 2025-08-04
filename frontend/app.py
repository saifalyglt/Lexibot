# Streamlit frontend for LexiBot legal document assistant
                with st.spinner("Processing document..."):
                    progress = st.progress(0)
                    for i in range(1, 101):
                        time.sleep(0.005)
                        progress.progress(i)
                    result = upload_document(uploaded_file.getvalue(), uploaded_file.name)
                    progress.empty()
                    if result:
                        st.success("✅ Document uploaded successfully!")
                        st.json(result)
                        st.session_state.uploaded_documents.append({
                            'name': uploaded_file.name,
                            'chunks': result.get('chunks_created', 0),
                            'timestamp': time.strftime('%Y-%m-%d %H:%M')
                        })
                        st.experimental_rerun()
        st.subheader("📚 Document Library")
        docs = get_document_list()
        if docs:
            for doc in docs:
                with st.expander(f"{doc['document_id']}"):
                    st.write(f"Chunks: {doc['chunk_count']}")
        else:
            st.info("No documents uploaded yet.")

    # Right: Search
    with col2:
        st.header("🔍 Legal Research")
        query = st.text_area("Enter your legal question:")
        top_k = st.selectbox("Results:", [3,5,7,10], index=1)
        if st.button("🚀 Search & Analyze", disabled=not query.strip()):
            with st.spinner("Analyzing..."):
                results = search_documents(query.strip(), top_k)
                if results and 'summary' in results:
                    st.markdown("### 🤖 AI-Generated Summary")
                    st.markdown(results['summary'])
                    if results.get('citations'):
                        st.markdown("### 📖 Source Citations")
                        for idx, cit in enumerate(results['citations'], 1):
                            with st.expander(f"Citation {idx} - {cit['document_id']}"):
                                st.write(f"Relevance: {cit['score']:.3f}")
                                st.write(f"> {cit['text']}")
                        st.session_state.search_history.insert(0, {
                            'query': query.strip(),
                            'timestamp': time.strftime('%H:%M:%S'),
                            'results_count': len(results.get('citations', []))
                        })
                else:
                    st.error("No summary generated.")

        # Search history
        if st.session_state.search_history:
            st.subheader("🕒 Recent Searches")
            for i, h in enumerate(st.session_state.search_history[:5]):
                with st.expander(f"[{h['timestamp']}] {h['query'][:50]}"):
                    st.write(f"Query: {h['query']}")
                    st.write(f"Results: {h['results_count']}")
                    if st.button("Repeat", key=f"rp_{i}"):
                        st.experimental_set_query_params(query=h['query'])
                        st.experimental_rerun()

if __name__ == '__main__':
    main()
