# Raw RAG: Retrieval-Augmented Generation from Scratch

## Introduction

Retrieval-Augmented Generation (RAG) is a powerful technique that enhances language models by retrieving relevant information from a knowledge base before generating responses. This repository demonstrates how to implement RAG using simple Python functions and libraries, without relying on complex frameworks like LangChain or LlamaIndex.

The goal of this project is to educate developers on building a RAG system from the ground up, providing a deeper understanding of the underlying processes and greater control over the implementation.

## Why "Raw" RAG?

While libraries like LangChain and LlamaIndex offer quick setup and implementation of RAG systems, there are several advantages to building a RAG pipeline from scratch:

1. **Deterministic Results**: By controlling each step of the process, you can ensure more consistent and reproducible outcomes.
2. **Enhanced Control**: Understanding and implementing each component allows for fine-tuned adjustments and optimizations.
3. **Reduced Dependencies**: Minimizing external libraries reduces potential security vulnerabilities and version conflicts.
4. **Transparency**: A "raw" approach makes the entire pipeline more inspectable and understandable.
5. **Customization**: Easily modify and extend the system to fit specific use cases without library constraints.
6. **Learning Opportunity**: Building from scratch provides invaluable insights into the RAG process and its components.

## Repository Contents

This repository contains several Jupyter notebooks demonstrating different aspects of building a RAG system:

1. [`raw_rag_01_basics.ipynb`](./raw_rag_01_basics.ipynb): Basic RAG implementation with embedding and retrieval
2. [`raw_rag_02_no_embed.ipynb`](./raw_rag_02_no_embed.ipynb): Retrieval techniques, such as BM25, without embeddings
3. [`raw_rag_03_clearer_ocr.ipynb`](./raw_rag_03_clearer_ocr.ipynb): Document preprocessing and OCR enhancement
4. [`raw_rag_04_summarize.ipynb`](./raw_rag_04_summarize.ipynb): Text summarization techniques
5. [`raw_rag_05_pydantic_is_all_you_need.ipynb`](./raw_rag_05_pydantic_is_all_you_need.ipynb): Implementing JSON parsing for structured output
6. [`raw_rag_06_metadata.ipynb`](./raw_rag_06_metadata.ipynb): Metadata extraction  
7. [`raw_rag_07_memory.ipynb`](./raw_rag_07_memory.ipynb): Memory implementation
8. [`raw_rag_08_evaluation.ipynb`](./raw_rag_08_evaluation.ipynb): Evaluation metrics for RAG

More to come ...

## Getting Started

### Prerequisites

- Python 3.9+
- Jupyter Notebook or JupyterLab

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/raw-rag.git
   cd raw-rag
   ```
   
2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate
   ```

3. Install the required dependencies, although you see `pip install` in the notebooks, it is recommended to install all dependencies at once:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the root directory and add the following environment variables needed for the notebooks:
   ```
   cp .env.example .env
   ```
   
5. Start Jupyter Notebook or JupyterLab:
   ```
   jupyter notebook
   ```

## Usage

Each notebook in the repository is self-contained and includes detailed explanations. To get started, open any notebook and run the cells sequentially. For example:

```python
# Example from raw_rag.ipynb
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def simple_retrieval(query_embedding, document_embeddings, k=5):
    similarities = cosine_similarity([query_embedding], document_embeddings)[0]
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    return top_k_indices, similarities[top_k_indices]

# Use this function in your RAG pipeline
```

## Benefits over Library-based Approaches

1. **Full Control**: Understand and modify every aspect of the RAG pipeline.
2. **Performance Optimization**: Fine-tune each component for your specific use case.
3. **Minimal Overhead**: Avoid unnecessary features and dependencies of larger libraries.
4. **Easy Debugging**: Quickly identify and fix issues in your implementation.
5. **Flexible Integration**: Easily incorporate the RAG system into existing projects.
6. **Educational Value**: Gain a deep understanding of RAG principles and implementation details.

## Todo

- [ ] Add additional advanced memory techniques
- [ ] Implement more agentic approaches to RAG
- [ ] More advanced document pro-processing techniques
- [ ] GraphRAG implementation
- [ ] If you have any suggestions, please open an issue or submit a pull request.

## Contributing

Contributions to this project are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch for your feature
3. Implement your changes
4. Write or update tests as necessary
5. Submit a pull request with a clear description of your changes

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- All the contributors to the RAG research and development
- The open-source community for providing valuable tools and libraries

---

Remember, while this "raw" approach offers many advantages, libraries like LangChain and LlamaIndex still have their place in rapid prototyping and development. The goal here is to provide an alternative that promotes understanding and control over the RAG process.
