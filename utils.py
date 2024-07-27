"""
Utility classes for Raw RAG
"""

import json
import math
import re
from collections import Counter
from heapq import nlargest
from string import punctuation
from typing import Any, Dict, List, Tuple, Callable, Optional, Type

import spacy
from spacy.language import Language
from pydantic import BaseModel
from pydantic.json_schema import model_json_schema
from spacy.lang.en.stop_words import STOP_WORDS
import nltk
from nltk.tokenize import word_tokenize


class TextProcessorError(Exception):
    """Custom exception for TextProcessor errors."""

    pass


class TextProcessor:
    def __init__(self, tokenizer: Callable[[str], List[str]] = word_tokenize):
        self.tokenizer: Callable[[str], List[str]] = tokenizer
        # Download NLTK data if not already present
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")

    def text_splitter(
        self,
        text: str,
        char_limit: int = 500,
        eos_characters: str = ".\n",
        overlap: int = 0,
    ) -> List[str]:
        """
        Splits a given text into paragraphs based on a character limit and end-of-sentence (EOS) characters.

        Args:
          text (str): The input text to be split into paragraphs.
          char_limit (int): The maximum character limit for each paragraph. Defaults to 500.
          eos_characters (str): The characters that mark the end of a sentence. Defaults to ".\n".
          overlap (int): The number of characters to overlap between paragraphs. Defaults to 0.

        Returns:
          List[str]: A list of paragraphs.

        Raises:
          TextProcessorError: If the input text is empty or if char_limit is less than 1.
        """
        if not text:
            raise TextProcessorError("Input text cannot be empty.")
        if char_limit < 1:
            raise TextProcessorError("Character limit must be at least 1.")
        if overlap < 0 or overlap >= char_limit:
            raise TextProcessorError(
                "Overlap must be non-negative and less than char_limit."
            )

        try:
            paragraphs = []
            current_paragraph = ""

            pattern = f"([{re.escape(eos_characters)}])"
            sentences = re.split(pattern, text)
            sentences = [
                "".join(sentences[i : i + 2]) for i in range(0, len(sentences), 2)
            ]

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                if len(current_paragraph) + len(sentence) <= char_limit:
                    current_paragraph += (
                        " " + sentence if current_paragraph else sentence
                    )
                else:
                    if current_paragraph:
                        paragraphs.append(current_paragraph.strip())
                    current_paragraph = sentence

                    # Check if we need to split the sentence further
                    while len(current_paragraph) > char_limit:
                        split_index = current_paragraph.rfind(" ", 0, char_limit)
                        if split_index == -1:
                            split_index = char_limit
                        paragraphs.append(current_paragraph[:split_index].strip())
                        current_paragraph = current_paragraph[split_index:].strip()

            if current_paragraph:
                paragraphs.append(current_paragraph.strip())

            # Add overlap while maintaining proper spacing
            if overlap > 0:
                overlapped_paragraphs = []
                for i, para in enumerate(paragraphs):
                    if i > 0:
                        overlap_text = paragraphs[i - 1][-overlap:].rstrip()
                        if (
                            overlap_text
                            and para
                            and not overlap_text[-1].isspace()
                            and not para[0].isspace()
                        ):
                            para = overlap_text + " " + para
                        else:
                            para = overlap_text + para
                    overlapped_paragraphs.append(para)
                return overlapped_paragraphs

            return paragraphs

        except Exception as e:
            raise TextProcessorError(f"Error splitting text: {str(e)}")
            

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize the input text into a list of tokens, removing punctuation.

        Args:
        text (str): The input text to tokenize.

        Returns:
        List[str]: A list of tokens without punctuation.

        Raises:
        TextProcessorError: If the input text is empty or not a string.
        """
        if not isinstance(text, str):
            raise TextProcessorError("Input must be a string.")
        if not text:
            raise TextProcessorError("Input text cannot be empty.")

        try:
            # Remove punctuation
            text_without_punct = re.sub(r'[^\w\s]', '', text)

            # Tokenize the text without punctuation
            tokens = self.tokenizer(text_without_punct)

            return tokens
        except Exception as e:
            raise TextProcessorError(f"Error tokenizing text: {str(e)}")


class KeywordExtractorError(Exception):
    """Custom exception for KeywordExtractor errors."""

    pass


class KeywordExtractor:

    def __init__(
        self, model: str = "en_core_web_sm", pos_tags: Optional[List[str]] = None
    ):
        self.nlp: Language = spacy.load(model)
        self.pos_tags: List[str] = pos_tags or ["PROPN", "ADJ", "NOUN", "VERB"]

    def extract_keywords(self, text: str, num_keywords: int = 5) -> Dict[str, Any]:
        """
        Extract dates, person names, locations, and keywords from the given text using spaCy.

        Args:
        text (str): The input text to process.
        num_keywords (int): The number of keywords to extract (default: 5).

        Returns:
        Dict[str, Any]: A dictionary containing extracted when, who, where, and keywords.

        Raises:
        KeywordExtractorError: If the input text is empty, not a string, or if num_keywords is less than 1.
        """
        if not isinstance(text, str):
            raise KeywordExtractorError("Input must be a string.")
        if not text:
            raise KeywordExtractorError("Input text cannot be empty.")
        if num_keywords < 1:
            raise KeywordExtractorError("Number of keywords must be at least 1.")

        try:
            doc = self.nlp(text)

            when, who, where = [], [], []
            for ent in doc.ents:
                if ent.label_ == "DATE":
                    when.append(ent.text)
                elif ent.label_ == "PERSON":
                    who.append(ent.text)
                elif ent.label_ == "GPE" or ent.label_ == "LOC":
                    where.append(ent.text)

            word_frequencies: Dict[str, float] = {}
            for token in doc:
                if (
                    token.pos_ in self.pos_tags
                    and not token.is_stop
                    and not token.is_punct
                ):
                    word_frequencies[token.text] = (
                        word_frequencies.get(token.text, 0) + 1
                    )

            keywords = []
            if word_frequencies:
                max_frequency = max(word_frequencies.values())
                for word in word_frequencies:
                    word_frequencies[word] = word_frequencies[word] / max_frequency
                keywords = sorted(
                    word_frequencies, key=word_frequencies.get, reverse=True
                )[:num_keywords]

            return {
                "when": list(set(when)),
                "who": list(set(who)),
                "where": list(set(where)),
                "keywords": keywords,
            }
        except Exception as e:
            raise KeywordExtractorError(f"Error extracting keywords: {str(e)}")


class SearchEngineError(Exception):
    """Custom exception for SearchEngine errors."""

    pass


class SearchEngine:

    def __init__(self, tokenizer: Callable[[str], List[str]] = str.split):
        self.tokenizer: Callable[[str], List[str]] = tokenizer

    def calculate_idf(self, corpus: List[List[str]]) -> Dict[str, float]:
        """
        Calculate the Inverse Document Frequency (IDF) for each term in the corpus.

        Args:
        corpus (List[List[str]]): A list of tokenized documents.

        Returns:
        Dict[str, float]: A dictionary mapping each term to its IDF value.

        Raises:
        SearchEngineError: If the corpus is empty or not in the correct format.
        """
        if (
            not corpus
            or not isinstance(corpus, list)
            or not all(isinstance(doc, list) for doc in corpus)
        ):
            raise SearchEngineError(
                "Corpus must be a non-empty list of tokenized documents (list of lists of strings)."
            )

        try:
            num_docs = len(corpus)
            idf: Dict[str, float] = {}

            for doc in corpus:
                for term in set(doc):
                    idf[term] = idf.get(term, 0) + 1

            for term, count in idf.items():
                idf[term] = math.log((num_docs - count + 0.5) / (count + 0.5) + 1)

            return idf
        except Exception as e:
            raise SearchEngineError(f"Error calculating IDF: {str(e)}")

    def bm25_search(
        self,
        query: str,
        documents: List[str],
        k: int = 5,
        b: float = 0.75,
        k1: float = 1.5,
        use_custom_tokenizer: bool = True
    ) -> List[Tuple[int, float]]:
        """
        Perform BM25 search on a list of documents given a query.

        Args:
        query (str): The search query.
        documents (List[str]): A list of documents to search through.
        k (int): The number of top results to return (default: 5).
        b (float): BM25 hyperparameter for length normalization (default: 0.75).
        k1 (float): BM25 hyperparameter for term frequency scaling (default: 1.5).
        use_custom_tokenizer (bool): Whether to use the custom tokenizer or default split (default: True).

        Returns:
        List[Tuple[int, float]]: A list of tuples containing the document index and its BM25 score,
                                 sorted in descending order by score.

        Raises:
        SearchEngineError: If the query is empty, documents list is empty, or if k, b, or k1 are invalid.
        """
        if not query:
            raise SearchEngineError("Search query cannot be empty.")
        if not documents:
            raise SearchEngineError("Documents list cannot be empty.")
        if k < 1:
            raise SearchEngineError("k must be at least 1.")
        if not 0 <= b <= 1:
            raise SearchEngineError("b must be between 0 and 1.")
        if k1 < 0:
            raise SearchEngineError("k1 must be non-negative.")

        try:
            tokenize = self.tokenizer if use_custom_tokenizer else str.split
            tokenized_query = tokenize(query)
            tokenized_docs = [tokenize(doc) for doc in documents]

            idf = self.calculate_idf(tokenized_docs)

            doc_lengths = [len(doc) for doc in tokenized_docs]
            avg_doc_length = sum(doc_lengths) / len(doc_lengths)

            scores = []
            for idx, doc in enumerate(tokenized_docs):
                score = 0
                doc_length = len(doc)
                term_frequencies = Counter(doc)

                for term in tokenized_query:
                    if term not in doc:
                        continue

                    tf = term_frequencies[term]
                    numerator = idf[term] * tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * doc_length / avg_doc_length)
                    score += numerator / denominator

                scores.append((idx, score))

            return sorted(scores, key=lambda x: x[1], reverse=True)[:k]
        except Exception as e:
            raise SearchEngineError(f"Error performing BM25 search: {str(e)}")


class SchemaConverterError(Exception):
    """Custom exception for SchemaConverter errors."""

    pass


class SchemaConverter:

    def __init__(
        self,
        custom_field_handlers: Optional[
            Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]]
        ] = None,
    ):
        self.custom_field_handlers: Dict[
            str, Callable[[Dict[str, Any]], Dict[str, Any]]
        ] = (custom_field_handlers or {})

    def pydantic_to_function_schema(
        self, model: Type[BaseModel], include_descriptions: bool = True
    ) -> Dict[str, Any]:
        """
        Converts a Pydantic model to a function schema compatible with OpenAI API.

        Args:
            model (type[BaseModel]): The Pydantic model to convert.
            include_descriptions (bool): Whether to include field descriptions in the schema (default: True).

        Returns:
            Dict[str, Any]: The function schema representing the Pydantic model.

        Raises:
            SchemaConverterError: If the input is not a valid Pydantic model.
        """
        if not isinstance(model, type) or not issubclass(model, BaseModel):
            raise SchemaConverterError("Input must be a valid Pydantic model class.")

        try:
            schema = model.model_json_schema()

            function_schema = {
                "type": "function",
                "function": {
                    "name": schema["title"].lower().replace(" ", "_"),
                    "description": schema.get("description", ""),
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": schema.get("required", []),
                    },
                },
            }

            for field_name, field_schema in schema["properties"].items():
                field_type, field_format = self._get_field_type_and_format(field_schema)

                if field_name in self.custom_field_handlers:
                    field_schema = self.custom_field_handlers[field_name](field_schema)

                field_info = {"type": field_type}

                if field_format:
                    field_info["format"] = field_format

                if include_descriptions and "description" in field_schema:
                    field_info["description"] = field_schema["description"]

                function_schema["function"]["parameters"]["properties"][field_name] = field_info

            return function_schema
        except Exception as e:
            raise SchemaConverterError(
                f"Error converting Pydantic model to function schema: {str(e)}"
            )

    def _get_field_type_and_format(self, field_schema: Dict[str, Any]) -> Tuple[str, Optional[str]]:
        """
        Helper method to determine the field type and format from the field schema.
        """
        if "type" in field_schema:
            field_type = field_schema["type"]
            field_format = field_schema.get("format")
        elif "$ref" in field_schema:
            field_type = "object"
            field_format = None
        elif "anyOf" in field_schema or "oneOf" in field_schema:
            # For union types, we'll default to string to ensure compatibility
            field_type = "string"
            field_format = None
        else:
            field_type = "string"  # Default to string if type can't be determined
            field_format = None

        # Ensure the field type is compatible with OpenAI API
        if field_type not in ["string", "number", "integer", "boolean", "object", "array"]:
            field_type = "string"  # Default to string for unsupported types

        return field_type, field_format


class MessageParserError(Exception):
    """Custom exception for MessageParser errors."""

    pass


class MessageParser:

    def __init__(
        self,
        custom_parsers: Optional[Dict[str, Callable[[str], Dict[str, Any]]]] = None,
    ):
        self.custom_parsers: Dict[str, Callable[[str], Dict[str, Any]]] = (
            custom_parsers or {}
        )

    def extract_and_parse_arguments(
        self, message: Any, parser: Optional[str] = None
    ) -> Dict[str, Any]:

        """
        Extracts and parses the arguments from the given message.

        Args:
            message: The message object containing tool calls.
            parser (Optional[str]): The name of a custom parser to use (if available).

        Returns:
            Dict[str, Any]: A dictionary representing the parsed arguments.

        Raises:
            MessageParserError: If the message object is invalid or if there's an error parsing the arguments.
        """
        if not hasattr(message, "tool_calls"):
            raise MessageParserError(
                "Invalid message object: missing 'tool_calls' attribute."
            )

        try:
            if message.tool_calls and len(message.tool_calls) > 0:
                first_tool_call = message.tool_calls[0]
                if not hasattr(first_tool_call, "function") or not hasattr(
                    first_tool_call.function, "arguments"
                ):
                    raise MessageParserError(
                        "Invalid tool call object: missing 'function' or 'arguments' attribute."
                    )

                arguments_str = first_tool_call.function.arguments

                if parser and parser in self.custom_parsers:
                    return self.custom_parsers[parser](arguments_str)
                else:
                    return json.loads(arguments_str)
            else:
                return {}
        except json.JSONDecodeError as e:
            raise MessageParserError(f"Error parsing JSON arguments: {str(e)}")
        except Exception as e:
            raise MessageParserError(
                f"Error extracting and parsing arguments: {str(e)}"
            )

    def add_custom_parser(
        self, name: str, parser_func: Callable[[str], Dict[str, Any]]
    ) -> None:
        """
        Adds a custom parser function to the MessageParser.

        Args:
            name (str): The name of the custom parser.
            parser_func (callable): The custom parser function.
        """
        self.custom_parsers[name] = parser_func


# Add this section at the bottom of the utils.py file

if __name__ == "__main__":
    print("Usage examples for Raw RAG utility classes with enhanced type hinting:")

    # TextProcessor examples
    print("\n1. TextProcessor examples:")
    from nltk.tokenize import word_tokenize

    custom_tokenizer: Callable[[str], List[str]] = lambda text: [
        token.lower() for token in word_tokenize(text) if token.isalnum()
    ]
    processor: TextProcessor = TextProcessor(tokenizer=custom_tokenizer)

    text: str = (
        "This is a long sentence that needs to be split. It contains multiple parts. Some parts might be very long and need further splitting to fit within the character limit."
    )

    print("Text splitting example with overlap:")
    split_text: List[str] = processor.text_splitter(text, char_limit=50, overlap=10)
    for i, paragraph in enumerate(split_text, 1):
        print(f"Paragraph {i}: {paragraph}")

    print("\nCustom tokenization example:")
    tokens: List[str] = processor.tokenize(
        "Hello World! This is a tokenization example."
    )
    print(f"Tokens: {tokens}")

    # KeywordExtractor examples
    print("\n2. KeywordExtractor examples:")
    custom_pos_tags: List[str] = ["NOUN", "PROPN", "ADJ"]
    extractor: KeywordExtractor = KeywordExtractor(
        model="en_core_web_sm", pos_tags=custom_pos_tags
    )

    sample_text: str = """
    On July 20, 1969, Neil Armstrong became the first person to walk on the Moon. 
    The Apollo 11 mission, commanded by Armstrong, was a major achievement for NASA and the United States.
    """
    keywords: Dict[str, Any] = extractor.extract_keywords(sample_text, num_keywords=5)
    print("Extracted keywords and entities:")
    print(f"Keywords: {keywords['keywords']}")
    print(f"Dates (When): {keywords['when']}")
    print(f"People (Who): {keywords['who']}")
    print(f"Locations (Where): {keywords['where']}")

    # SearchEngine examples
    print("\n3. SearchEngine examples:")
    custom_tokenizer: Callable[[str], List[str]] = lambda text: text.lower().split()
    search_engine: SearchEngine = SearchEngine(tokenizer=custom_tokenizer)

    documents: List[str] = [
        "The quick brown fox jumps over the lazy dog",
        "A quick brown dog outpaces a fast fox",
        "The lazy dog sleeps all day",
        "A fast fox is quicker than a lazy dog",
    ]
    query: str = "quick brown fox"
    results: List[Tuple[int, float]] = search_engine.bm25_search(
        query, documents, k=2, b=0.8, k1=1.8
    )
    print(f"Top 2 results for query '{query}':")
    for doc_id, score in results:
        print(f"Document {doc_id}: '{documents[doc_id]}' (Score: {score:.4f})")

    # SchemaConverter examples
    print("\n4. SchemaConverter examples:")
    from pydantic import BaseModel, Field

    def custom_field_handler(field_schema: Dict[str, Any]) -> Dict[str, Any]:
        field_schema["description"] = f"Custom: {field_schema.get('description', '')}"
        return field_schema

    converter: SchemaConverter = SchemaConverter(
        custom_field_handlers={"name": custom_field_handler}
    )

    class UserModel(BaseModel):
        id: int = Field(..., description="The user's ID")
        name: str = Field(..., description="The user's name")
        email: str = Field(..., description="The user's email address")

    schema: Dict[str, Any] = converter.pydantic_to_function_schema(
        UserModel, include_descriptions=True
    )
    print("Converted Pydantic model to function schema:")
    print(json.dumps(schema, indent=2))

    # MessageParser examples
    print("\n5. MessageParser examples:")
    from collections import namedtuple

    def custom_parser(arguments_str: str) -> Dict[str, Any]:
        parsed = json.loads(arguments_str)
        parsed["custom_field"] = "This is a custom parsed field"
        return parsed

    parser: MessageParser = MessageParser(custom_parsers={"custom": custom_parser})

    MockToolCall = namedtuple("MockToolCall", ["function"])
    MockFunction = namedtuple("MockFunction", ["arguments"])
    mock_message = namedtuple("MockMessage", ["tool_calls"])

    tool_call = MockToolCall(MockFunction('{"arg1": "value1", "arg2": 42}'))
    message = mock_message([tool_call])

    parsed_args: Dict[str, Any] = parser.extract_and_parse_arguments(
        message, parser="custom"
    )
    print("Parsed arguments from message (with custom parser):")
    print(parsed_args)
