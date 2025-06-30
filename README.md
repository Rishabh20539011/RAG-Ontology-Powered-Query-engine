# RAG-Powered Ontology Builder 🧠➡️📊

A sophisticated Retrieval-Augmented Generation (RAG) system that automatically builds knowledge graphs and ontologies from your documents. This project combines the power of LLMs with Neo4j graph database to create interactive, queryable knowledge representations.

## 🚀 What This Project Does

This application takes your text documents (PDFs, TXT, Markdown files, or ZIP archives) and:

1. **Extracts Knowledge**: Uses advanced LLMs to identify entities, relationships, and concepts
2. **Builds Knowledge Graphs**: Creates structured ontologies stored in Neo4j database
3. **Enables Intelligent Querying**: Provides a natural language interface to query your knowledge base
4. **Visualizes Relationships**: Generates interactive graph visualizations
5. **Supports RAG Queries**: Answers questions by retrieving relevant graph context

## 🏗️ Architecture

- **Frontend**: Streamlit web interface for document upload and querying
- **LLM**: Groq API with Llama 3.3 70B model for knowledge extraction
- **Embeddings**: Sentence Transformers for semantic similarity
- **Graph Database**: Neo4j for persistent knowledge storage
- **RAG Engine**: LlamaIndex for document processing and retrieval

## 📋 Prerequisites

- Python 3.8+
- Neo4j Database (local or remote)
- Groq API key

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd rag-ontology-builder
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Neo4j Database

#### Option A: Local Installation
1. Download and install [Neo4j Desktop](https://neo4j.com/download/)
2. Create a new database with:
   - Username: `neo4j`
   - Password: `your_password`
   - Bolt URL: `bolt://localhost:7687`

#### Option B: Neo4j AuraDB (Cloud)
1. Create a free account at [Neo4j Aura](https://neo4j.com/cloud/aura/)
2. Create a new database instance
3. Note the connection URL, username, and password

### 4. Configure API Keys and Settings

Edit `config.yaml` with your credentials:

```yaml
# Neo4j Database Configuration
neo4j:
  url: bolt://localhost:7687        # Update for your Neo4j instance
  username: neo4j
  password: "your_neo4j_password"   # Replace with your password

# Groq API Configuration
groq:
  api_key: "your_groq_api_key"      # Get from https://console.groq.com/

# Embedding Model Configuration
embedding:
  model_name: sentence-transformers/all-MiniLM-L6-v2

# LLM Configuration
llm:
  model_name: llama-3.3-70b-versatile
```

### 5. Get Your Groq API Key

1. Visit [Groq Console](https://console.groq.com/)
2. Sign up for a free account
3. Generate an API key
4. Add the key to your `config.yaml`

## 🚀 Usage

### 1. Start the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### 2. Upload Documents

- Click "Choose file(s)" to upload:
  - Individual files: `.txt`, `.md`, `.pdf`
  - ZIP archives containing multiple documents
- Click "Build Ontology" to start processing

### 3. View and Query Your Knowledge Graph

- The system will build your ontology (this may take a few minutes)
- View the generated graph visualization
- Use natural language to query your knowledge base
- Explore retrieved graph triplets for transparency

### 4. Access Neo4j Browser (Optional)

Visit `http://localhost:7474` to explore your knowledge graph interactively in Neo4j Browser.

## 📁 Project Structure

```
rag-ontology-builder/
├── app.py                      # Main Streamlit application
├── config.yaml                 # Configuration file (API keys, database settings)
├── config_loader.py           # Configuration loader utility
├── ontology_builder.py        # Core ontology building logic
├── pride_prejudice_chapters.py # Example data preparation script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🔧 Configuration Details

### Models Used

- **LLM**: Llama 3.3 70B Versatile (via Groq)
  - Used for entity extraction and relationship identification
  - Provides natural language query understanding
  
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
  - Generates semantic embeddings for similarity search
  - Enables context-aware retrieval

### Graph Storage

- **Primary**: Neo4j PropertyGraph for persistent storage
- **Secondary**: In-memory NetworkX graph for visualization

## 🔍 Features

### Document Processing
- Multi-format support (PDF, TXT, Markdown)
- Batch processing via ZIP uploads
- Automatic text extraction and chunking

### Knowledge Extraction
- Entity recognition and classification
- Relationship extraction between entities
- Concept mapping and hierarchy building

### Intelligent Querying
- Natural language question answering
- Semantic similarity retrieval
- Graph context integration
- Transparent result explanation

### Visualization
- Interactive graph visualizations
- Neo4j Browser integration
- Export capabilities

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Resources

- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [Neo4j Graph Database](https://neo4j.com/)
- [Groq API Documentation](https://console.groq.com/docs)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 🆘 Support

If you encounter issues:

1. Check that Neo4j is running and accessible
2. Verify your API keys in `config.yaml`
3. Ensure all dependencies are installed correctly
4. Check the Streamlit logs for detailed error messages

## 🎯 Example Use Cases

- **Research**: Build knowledge graphs from academic papers
- **Documentation**: Create queryable knowledge bases from technical docs
- **Business Intelligence**: Extract insights from reports and presentations
- **Education**: Build interactive learning resources from textbooks

---

**Happy Knowledge Graphing!** 🎉
