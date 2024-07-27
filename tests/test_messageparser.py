import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
from collections import namedtuple

from utils import MessageParser, MessageParserError


class TestMessageParser(unittest.TestCase):
    def setUp(self):
        self.parser = MessageParser()

    def test_extract_and_parse_arguments_normal(self):
        MockToolCall = namedtuple("MockToolCall", ["function"])
        MockFunction = namedtuple("MockFunction", ["arguments"])
        mock_message = namedtuple("MockMessage", ["tool_calls"])

        tool_call = MockToolCall(MockFunction('{"arg1": "value1", "arg2": 42}'))
        message = mock_message([tool_call])

        result = self.parser.extract_and_parse_arguments(message)
        self.assertEqual(result, {"arg1": "value1", "arg2": 42})

    def test_extract_and_parse_arguments_custom_parser(self):
        def custom_parser(arguments_str):
            parsed = eval(arguments_str)
            parsed["custom_field"] = "Custom value"
            return parsed

        parser = MessageParser(custom_parsers={"custom": custom_parser})

        MockToolCall = namedtuple("MockToolCall", ["function"])
        MockFunction = namedtuple("MockFunction", ["arguments"])
        mock_message = namedtuple("MockMessage", ["tool_calls"])

        tool_call = MockToolCall(MockFunction('{"arg1": "value1", "arg2": 42}'))
        message = mock_message([tool_call])

        result = parser.extract_and_parse_arguments(message, parser="custom")
        self.assertEqual(
            result, {"arg1": "value1", "arg2": 42, "custom_field": "Custom value"}
        )

    def test_extract_and_parse_arguments_invalid_message(self):
        with self.assertRaises(MessageParserError):
            self.parser.extract_and_parse_arguments({})  # Invalid message object

    def test_extract_and_parse_arguments_invalid_json(self):
        MockToolCall = namedtuple("MockToolCall", ["function"])
        MockFunction = namedtuple("MockFunction", ["arguments"])
        mock_message = namedtuple("MockMessage", ["tool_calls"])

        tool_call = MockToolCall(MockFunction('{"invalid_json":}'))
        message = mock_message([tool_call])

        with self.assertRaises(MessageParserError):
            self.parser.extract_and_parse_arguments(message)

    def test_add_custom_parser(self):
        def new_parser(arguments_str):
            return {"parsed": arguments_str}

        self.parser.add_custom_parser("new", new_parser)

        MockToolCall = namedtuple("MockToolCall", ["function"])
        MockFunction = namedtuple("MockFunction", ["arguments"])
        mock_message = namedtuple("MockMessage", ["tool_calls"])

        tool_call = MockToolCall(MockFunction("test"))
        message = mock_message([tool_call])

        result = self.parser.extract_and_parse_arguments(message, parser="new")
        self.assertEqual(result, {"parsed": "test"})


if __name__ == "__main__":
    unittest.main()
    unittest.main()
