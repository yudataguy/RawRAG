# Raw RAG Utility Classes

This library provides a set of utility classes for text processing, keyword extraction, search functionality, schema conversion, and message parsing. Below you'll find detailed explanations and examples for each class.

## Table of Contents

1. [TextProcessor](#textprocessor)
2. [KeywordExtractor](#keywordextractor)
3. [SearchEngine](#searchengine)
4. [SchemaConverter](#schemaconverter)
5. [MessageParser](#messageparser)

## TextProcessor

The TextProcessor class provides methods for text splitting and tokenization.

### Methods

#### `__init__(self, tokenizer: Callable[[str], List[str]] = word_tokenize)`

Initializes the TextProcessor with a custom tokenizer (default is NLTK's word_tokenize).

#### `text_splitter(self, text: str, char_limit: int = 500, eos_characters: str = ".\n", overlap: int = 0) -> List[str]`

Splits the input text into paragraphs based on character limit and end-of-sentence characters.

#### `tokenize(self, text: str) -> List[str]`

Tokenizes the input text using the specified tokenizer.

### Example Usage

```python
from nltk.tokenize import word_tokenize

# Initialize TextProcessor with a custom tokenizer
custom_tokenizer = lambda text: [token.lower() for token in word_tokenize(text) if token.isalnum()]
processor = TextProcessor(tokenizer=custom_tokenizer)

# Text splitting
text = "This is a long sentence. It needs to be split into smaller parts."
split_text = processor.text_splitter(text, char_limit=30, overlap=5)
print(split_text)
# Output: ['This is a long sentence.', 'ence. It needs to be split', 'split into smaller parts.']

# Tokenization
tokens = processor.tokenize("Hello, world! This is an example.")
print(tokens)
# Output: ['hello', 'world', 'this', 'is', 'an', 'example']
```

## KeywordExtractor

The KeywordExtractor class uses spaCy to extract keywords, dates, person names, and locations from text.

### Methods

#### `__init__(self, model: str = "en_core_web_sm", pos_tags: Optional[List[str]] = None)`

Initializes the KeywordExtractor with a specified spaCy model and POS tags for keyword extraction.

#### `extract_keywords(self, text: str, num_keywords: int = 5) -> Dict[str, Any]`

Extracts keywords, dates, person names, and locations from the input text.

### Example Usage

```python
# Initialize KeywordExtractor with custom POS tags
extractor = KeywordExtractor(model="en_core_web_sm", pos_tags=["NOUN", "PROPN", "ADJ"])

# Extract keywords and entities
text = "On July 20, 1969, Neil Armstrong walked on the Moon. NASA achieved this milestone."
result = extractor.extract_keywords(text, num_keywords=3)
print(result)
# Output: {
#     'keywords': ['Moon', 'Neil Armstrong', 'NASA'],
#     'when': ['July 20, 1969'],
#     'who': ['Neil Armstrong'],
#     'where': ['Moon']
# }
```

## SearchEngine

The SearchEngine class implements BM25 search functionality for document retrieval.

### Methods

#### `__init__(self, tokenizer: Callable[[str], List[str]] = str.split)`

Initializes the SearchEngine with a custom tokenizer (default is str.split).

#### `calculate_idf(self, corpus: List[List[str]]) -> Dict[str, float]`

Calculates the Inverse Document Frequency (IDF) for terms in the corpus.

#### `bm25_search(self, query: str, documents: List[str], k: int = 5, b: float = 0.75, k1: float = 1.5, use_custom_tokenizer: bool = True) -> List[Tuple[int, float]]`

Performs BM25 search on the documents given a query.

### Example Usage

```python
# Initialize SearchEngine with a custom tokenizer
custom_tokenizer = lambda text: text.lower().split()
search_engine = SearchEngine(tokenizer=custom_tokenizer)

# Perform BM25 search
documents = [
    "The quick brown fox jumps over the lazy dog",
    "A quick brown dog outpaces a fast fox",
    "The lazy dog sleeps all day"
]
query = "quick brown fox"
results = search_engine.bm25_search(query, documents, k=2)
print(results)
# Output: [(0, 0.744), (1, 0.372)]  # (document_index, score)
```

## SchemaConverter

The SchemaConverter class converts Pydantic models to function schemas.

### Methods

#### `__init__(self, custom_field_handlers: Optional[Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]]] = None)`

Initializes the SchemaConverter with optional custom field handlers.

#### `pydantic_to_function_schema(self, model: Type[BaseModel], include_descriptions: bool = True) -> Dict[str, Any]`

Converts a Pydantic model to a function schema.

### Example Usage

```python
from pydantic import BaseModel, Field

# Define a custom field handler
def custom_field_handler(field_schema):
    field_schema["description"] = f"Custom: {field_schema.get('description', '')}"
    return field_schema

# Initialize SchemaConverter with the custom field handler
converter = SchemaConverter(custom_field_handlers={"name": custom_field_handler})

# Define a Pydantic model
class UserModel(BaseModel):
    id: int = Field(..., description="The user's ID")
    name: str = Field(..., description="The user's name")
    email: str = Field(..., description="The user's email address")

# Convert the model to a function schema
schema = converter.pydantic_to_function_schema(UserModel)
print(json.dumps(schema, indent=2))
# Output: A JSON schema with custom description for the "name" field
```

## MessageParser

The MessageParser class extracts and parses arguments from message objects.

### Methods

#### `__init__(self, custom_parsers: Optional[Dict[str, Callable[[str], Dict[str, Any]]]] = None)`

Initializes the MessageParser with optional custom parsers.

#### `extract_and_parse_arguments(self, message: Any, parser: Optional[str] = None) -> Dict[str, Any]`

Extracts and parses arguments from the given message.

#### `add_custom_parser(self, name: str, parser_func: Callable[[str], Dict[str, Any]]) -> None`

Adds a custom parser function to the MessageParser.

### Example Usage

```python
# Define a custom parser
def custom_parser(arguments_str):
    parsed = json.loads(arguments_str)
    parsed["custom_field"] = "This is a custom parsed field"
    return parsed

# Initialize MessageParser with the custom parser
parser = MessageParser(custom_parsers={"custom": custom_parser})

# Mock a message object (for demonstration purposes)
from collections import namedtuple
MockToolCall = namedtuple('MockToolCall', ['function'])
MockFunction = namedtuple('MockFunction', ['arguments'])
mock_message = namedtuple('MockMessage', ['tool_calls'])

tool_call = MockToolCall(MockFunction('{"arg1": "value1", "arg2": 42}'))
message = mock_message([tool_call])

# Extract and parse arguments using the custom parser
parsed_args = parser.extract_and_parse_arguments(message, parser="custom")
print(parsed_args)
# Output: {'arg1': 'value1', 'arg2': 42, 'custom_field': 'This is a custom parsed field'}
```

This README provides an overview of each class, its methods, and example usage. Users can refer to this documentation to understand how to effectively use the Raw RAG utility classes in their projects.
