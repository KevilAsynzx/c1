# 1. First, let's enhance the vector store initialization with hybrid search capabilities

def initialize_vector_store():
    """Initialize the vector store with real estate knowledge base with hybrid search capabilities"""
    global vector_store
    
    try:
        # Check if we have a persisted vector store
        if os.path.exists("vector_store"):
            logger.info("Loading existing vector store...")
            vector_store = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)
            return
        
        # If not, create a new one from the knowledge base
        logger.info("Creating new vector store from knowledge base...")
        with open('data/real_estate_knowledge.txt', 'r') as f:
            real_estate_knowledge = f.read()
        
        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Smaller chunks for more precise retrieval
            chunk_overlap=100
        )
        chunks = text_splitter.split_text(real_estate_knowledge)
        
        # Extract section headers to categorize content
        import re
        section_pattern = r'#\s+(.*?)\n'
        sections = re.findall(section_pattern, real_estate_knowledge)
        logger.info(f"Found {len(sections)} sections in knowledge base")
        
        # Create a mapping of content to its section
        section_mapping = {}
        for i, section in enumerate(sections):
            # Get content between this section and the next
            start_pattern = f"# {section}\n"
            start_idx = real_estate_knowledge.find(start_pattern) + len(start_pattern)
            
            if i < len(sections) - 1:
                end_pattern = f"# {sections[i+1]}\n"
                end_idx = real_estate_knowledge.find(end_pattern)
            else:
                end_idx = len(real_estate_knowledge)
            
            section_content = real_estate_knowledge[start_idx:end_idx].strip()
            
            # Map section content to its name
            section_mapping[section_content] = section
            
            logger.info(f"Section '{section}' contains {len(section_content)} characters")
        
        # Add metadata to each chunk to track its source
        documents = []
        for i, chunk in enumerate(chunks):
            # Determine which section this chunk belongs to
            chunk_section = "Unknown"
            for section_content, section_name in section_mapping.items():
                if chunk in section_content:
                    chunk_section = section_name
                    break
            
            # Clean section name for use as a category
            clean_section = chunk_section.lower().replace(' ', '_')
            
            # Extract keywords for hybrid search using RAKE algorithm
            from rake_nltk import Rake
            r = Rake()
            r.extract_keywords_from_text(chunk)
            # Get top 5 keywords
            keywords = [kw for kw, score in r.get_ranked_phrases_with_scores()[:5]]
            
            # Add metadata to track the source, section, and keywords
            metadata = {
                'source': 'real_estate_knowledge.txt',
                'chunk_id': i,
                'section': clean_section,
                'keywords': ",".join(keywords)
            }
            documents.append({'content': chunk, 'metadata': metadata})
            
            logger.info(f"Chunk {i} assigned to section '{chunk_section}' with keywords: {keywords}")
        
        # Create vector store with metadata
        vector_store = FAISS.from_texts([doc['content'] for doc in documents], embeddings, metadatas=[doc['metadata'] for doc in documents])
        
        # Save the vector store for future use
        vector_store.save_local("vector_store")
        
    except Exception as e:
        logger.error(f"Error initializing vector store: {e}")
        # Create an empty vector store if there's an error
        vector_store = FAISS.from_texts(["Real estate chatbot information"], embeddings)


# 2. Now let's create a hybrid search retriever function that combines semantic and keyword search

from langchain.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def create_hybrid_retriever(vector_store):
    """Create a hybrid retriever that combines semantic search with keyword-based search"""
    
    # Extract texts from vector store for BM25
    docs = [doc for doc in vector_store.docstore._dict.values()]
    texts = [doc.page_content for doc in docs]
    
    # Create BM25 retriever for keyword-based search
    bm25_retriever = BM25Retriever.from_texts(texts)
    bm25_retriever.k = 5
    
    # Create vector store retriever for semantic search
    faiss_retriever = vector_store.as_retriever(
        search_type="mmr",  # Use Maximum Marginal Relevance for more diverse context
        search_kwargs={
            "k": 5,  # Retrieve more documents for better context
            "fetch_k": 10,  # Consider a larger initial set before filtering down
            "lambda_mult": 0.7  # Balance between relevance and diversity
        }
    )
    
    # Create an ensemble retriever that combines both approaches
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.3, 0.7]  # Weights for keyword vs semantic search
    )
    
    return ensemble_retriever


# 3. Let's enhance the hallucination risk evaluation function with embedding-based similarity scoring

def evaluate_hallucination_risk(query, response, source_docs, embeddings_model=None):
    """
    Enhanced evaluation of hallucination risk using embedding-based similarity
    alongside traditional methods.
    """
    if embeddings_model is None:
        embeddings_model = OpenAIEmbeddings()
    
    issues = []
    
    # 1. Check if response contains information not in source docs (word-based)
    response_words = set(response.lower().split())
    source_text = " ".join([doc.page_content for doc in source_docs]).lower()
    source_words = set(source_text.split())
    
    # Find words in response that aren't in sources (excluding common words)
    common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", 
                   "about", "is", "are", "be", "as", "it", "this", "that", "by", "from", "have", 
                   "has", "had", "of", "you", "your", "can", "will", "i", "don't", "if", "which"}
    
    unique_words = [word for word in response_words if word not in source_words and word not in common_words]
    
    # Count "new" terms that aren't in the sources
    new_term_count = len(unique_words)
    if new_term_count > 5:
        issues.append(f"Response contains {new_term_count} terms not found in source documents")
    
    # 2. Embedding-based similarity check
    # Split response into sentences for more fine-grained analysis
    import re
    response_sentences = re.split(r'(?<=[.!?])\s+', response)
    
    # Get embeddings for source documents and response sentences
    try:
        source_embeddings = embeddings_model.embed_documents([doc.page_content for doc in source_docs])
        response_embeddings = embeddings_model.embed_documents(response_sentences)
        
        # Calculate similarity scores for each response sentence against all source documents
        sentence_scores = []
        unsupported_sentences = []
        
        for i, resp_emb in enumerate(response_embeddings):
            # Reshape for sklearn cosine similarity
            resp_emb_reshaped = np.array(resp_emb).reshape(1, -1)
            source_emb_reshaped = np.array(source_embeddings)
            
            # Calculate max similarity score against any source document
            similarities = cosine_similarity(resp_emb_reshaped, source_emb_reshaped)[0]
            max_similarity = np.max(similarities)
            sentence_scores.append(max_similarity)
            
            # Flag sentences with low similarity to any source
            if max_similarity < 0.75 and len(response_sentences[i].split()) > 5:  # Ignore short sentences
                unsupported_sentences.append({
                    'sentence': response_sentences[i],
                    'similarity': max_similarity
                })
        
        # Add issue for sentences with low similarity to any source
        if unsupported_sentences:
            issues.append(f"Found {len(unsupported_sentences)} sentences with low similarity to source material")
            for sent in unsupported_sentences[:3]:  # Only show top 3 examples
                issues.append(f"Low similarity ({sent['similarity']:.2f}): '{sent['sentence']}'")
            
        # Calculate overall semantic similarity score
        avg_similarity = np.mean(sentence_scores)
        if avg_similarity < 0.8:
            issues.append(f"Overall semantic similarity to sources is low: {avg_similarity:.2f}")
    
    except Exception as e:
        logger.error(f"Error in embedding similarity calculation: {e}")
        # If embedding-based similarity fails, fall back to traditional methods
    
    # 3. Check for specific content markers that often indicate hallucination
    hallucination_markers = [
        "studies show", "research indicates", "experts say", "it is well known",
        "statistics show", "according to experts", "generally speaking", "typically",
        "in most cases", "it's common practice", "most professionals"
    ]
    
    for marker in hallucination_markers:
        if marker in response.lower():
            issues.append(f"Response contains potential hallucination marker: '{marker}'")
    
    # 4. Check if the response is much longer than the source material
    if len(response) > len(source_text) * 1.5:
        issues.append("Response is significantly longer than source material")
    
    # 5. Check for specific numbers or statistics not in sources
    import re
    response_stats = re.findall(r'\d+%|\d+\.\d+|\$\d+|\d+ percent', response.lower())
    source_stats = re.findall(r'\d+%|\d+\.\d+|\$\d+|\d+ percent', source_text)
    
    new_stats = [stat for stat in response_stats if stat not in source_stats]
    if new_stats:
        issues.append(f"Response contains statistics not found in sources: {new_stats}")
    
    # Calculate overall risk score (with embedding similarity if available)
    try:
        # Include semantic similarity in risk score (25% of the weight)
        similarity_factor = (1 - avg_similarity) * 25  # Lower similarity = higher risk
    except:
        similarity_factor = 0
    
    # Base risk on issues and new terms
    base_risk = min(75, (len(issues) * 15) + (new_term_count * 1.5))
    
    # Combine for final risk score (cap at 100)
    risk_score = min(100, base_risk + similarity_factor)
    
    # Return structured results
    return {
        "risk_score": risk_score,
        "issues": issues,
        "new_terms": unique_words[:10],  # Return at most 10 examples
        "semantic_similarity": avg_similarity if 'avg_similarity' in locals() else None,
        "unsupported_sentences": unsupported_sentences[:3] if 'unsupported_sentences' in locals() else []
    }


# 4. Finally, let's update the chat endpoint to use our new hybrid retriever and enhanced hallucination detection

@app.route('/api/chat', methods=['POST'])
@token_required
def chat():
    """Handle chat queries using hybrid search retrieval and enhanced hallucination detection"""
    if not rate_limit():
        return jsonify({'message': 'Rate limit exceeded'}), 429
    
    data = request.get_json()
    query = data.get('query', '')
    
    if not query:
        return jsonify({'message': 'Query is required'}), 400
    
    # Check if query exactly matches a predefined question
    for question in PREDEFINED_QUESTIONS:
        if query.lower() == question['text'].lower() and question['id'] in PREDEFINED_RESPONSES:
            logger.info(f"Using predefined response for question ID: {question['id']}")
            return jsonify({
                'response': PREDEFINED_RESPONSES[question['id']],
                'source': 'predefined'
            })
    
    # Try to find a similar predefined question
    for question in PREDEFINED_QUESTIONS:
        if query.lower() in question['text'].lower() and question['id'] in PREDEFINED_RESPONSES:
            logger.info(f"Using similar predefined response for question ID: {question['id']}")
            return jsonify({
                'response': PREDEFINED_RESPONSES[question['id']],
                'source': 'similar_predefined'
            })
    
    # If no predefined response, use hybrid retrieval if vector store is available
    if vector_store:
        try:
            # Create a hybrid retriever combining semantic and keyword search
            retriever = create_hybrid_retriever(vector_store)
            
            # Create a template for real estate queries with strict anti-hallucination guardrails
            template = """
            You are a helpful real estate assistant. Your task is to answer the question using ONLY the information provided in the context below. 
            
            STRICT GUIDELINES:
            1. ONLY use information explicitly stated in the provided context pieces.
            2. If the context doesn't contain the information needed to answer the question fully, say "I don't have enough information to answer that question completely" and then share what you can based ONLY on the context provided.
            3. NEVER add any information, examples, or details that aren't explicitly in the context, even if they seem obvious or helpful.
            4. Do not elaborate beyond what is directly supported by the context.
            5. If the question is completely unrelated to the context, respond with "I don't have information about that topic in my knowledge base."
            6. Format your answer in a clear, concise manner.
            
            Context:
            {context}
            
            Question: {question}
            Answer (using ONLY information from the context):
            """
            prompt = PromptTemplate(
                template=template,
                input_variables=["context", "question"]
            )
            
            # Create a chain with stricter parameters to reduce hallucinations
            llm = ChatOpenAI(
                model_name="gpt-3.5-turbo", 
                temperature=0,  # Use temperature=0 to reduce creativity/hallucinations
            )
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={
                    "prompt": prompt,
                    "verbose": True  # This helps with debugging
                }
            )
            
            # Get response from RAG
            result = qa_chain({"query": query})
            logger.info("Using hybrid retrieval for response")
            
            # Extract the source documents for verification
            source_docs = result.get('source_documents', [])
            
            # Check if the response is based on retrieved documents
            if not source_docs:
                logger.warning("RAG returned no source documents - potential hallucination")
                # Fall back to a generic response rather than potentially hallucinated content
                return jsonify({
                    'response': "I don't have specific information about that in my knowledge base. Would you like to know about something else related to real estate?",
                    'source': 'rag_no_sources'
                })
            
            # Verify that the response is grounded in the sources using enhanced evaluation
            response_content = result['result']
            
            # Use enhanced hallucination evaluation with embedding-based similarity
            hallucination_eval = evaluate_hallucination_risk(
                query, 
                response_content, 
                source_docs,
                embeddings_model=embeddings
            )
            
            logger.info(f"Hallucination risk score: {hallucination_eval['risk_score']}")
            
            # Include semantic similarity in response for monitoring
            semantic_similarity = hallucination_eval.get('semantic_similarity')
            if semantic_similarity:
                logger.info(f"Semantic similarity score: {semantic_similarity:.4f}")
            
            # If high hallucination risk detected, modify the response
            if hallucination_eval['risk_score'] > 50:
                logger.warning(f"High hallucination risk detected: {hallucination_eval['issues']}")
                
                # Check if there are specific problematic sentences
                unsupported_sentences = hallucination_eval.get('unsupported_sentences', [])
                if unsupported_sentences:
                    logger.warning(f"Unsupported sentences detected: {unsupported_sentences}")
                
                # Generate a better fallback response that combines direct quotes
                # from multiple relevant sources (not just the first one)
                source_quotes = []
                for idx, doc in enumerate(source_docs[:3]):  # Use top 3 sources
                    source_quotes.append(f"Source {idx+1}: \"{doc.page_content.strip()}\"")
                
                fallback_response = (
                    "I can only provide information based on what I know. " + 
                    "Here's what my knowledge base says about this topic:\n\n" + 
                    "\n\n".join(source_quotes)
                )
                
                return jsonify({
                    'response': fallback_response,
                    'source': 'rag_fallback',
                    'confidence': 'low',
                    'hallucination_risk': 'high',
                    'semantic_similarity': semantic_similarity
                })
            
            # Check the confidence - if response contains uncertainty markers added by our prompt
            uncertain_phrases = [
                "I don't have enough information",
                "I don't have information",
                "not mentioned in the context",
                "not provided in the context",
                "not specified in the"
            ]
            
            is_uncertain = any(phrase in response_content for phrase in uncertain_phrases)
            
            # Log the confidence level for monitoring
            if is_uncertain:
                logger.info("RAG returned low-confidence response")
            
            return jsonify({
                'response': response_content,
                'source': 'hybrid_rag',
                'confidence': 'low' if is_uncertain else 'high',
                'hallucination_risk': 'low' if hallucination_eval['risk_score'] < 30 else 'medium',
                'semantic_similarity': semantic_similarity
            })
        except Exception as e:
            logger.error(f"Error using hybrid retrieval: {e}")
            # Fall back to OpenAI if retrieval fails
    
    # If no vector store or retrieval failed, use direct OpenAI call
    try:
        logger.info("Using direct OpenAI call")
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful real estate assistant. Provide concise and accurate information about real estate topics."},
                {"role": "user", "content": query}
            ],
            max_tokens=150  # Limit token usage
        )
        
        response = completion.choices[0].message.content
        
        return jsonify({
            'response': response,
            'source': 'openai'
        })
    except Exception as e:
        logger.error(f"Error with OpenAI: {e}")
        return jsonify({'message': 'Error processing request'}), 500

