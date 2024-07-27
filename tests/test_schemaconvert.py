import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest

from pydantic import BaseModel, Field

from utils import SchemaConverter, SchemaConverterError


class TestSchemaConverter(unittest.TestCase):
    def setUp(self):
        self.converter = SchemaConverter()

    def test_pydantic_to_function_schema_normal(self):
        class TestModel(BaseModel):
            id: int = Field(..., description="The ID")
            name: str = Field(..., description="The name")

        schema = self.converter.pydantic_to_function_schema(TestModel)
        self.assertIn("type", schema)
        self.assertIn("function", schema)
        self.assertIn("parameters", schema["function"])
        self.assertIn("properties", schema["function"]["parameters"])
        self.assertIn("id", schema["function"]["parameters"]["properties"])
        self.assertIn("name", schema["function"]["parameters"]["properties"])

    def test_pydantic_to_function_schema_custom_handler(self):
        def custom_handler(field_schema):
            field_schema["description"] = "Custom: " + field_schema.get(
                "description", ""
            )
            return field_schema

        converter = SchemaConverter(custom_field_handlers={"name": custom_handler})

        class TestModel(BaseModel):
            id: int = Field(..., description="The ID")
            name: str = Field(..., description="The name")

        schema = converter.pydantic_to_function_schema(TestModel)
        self.assertTrue(
            schema["function"]["parameters"]["properties"]["name"][
                "description"
            ].startswith("Custom:")
        )

    def test_pydantic_to_function_schema_invalid_input(self):
        with self.assertRaises(SchemaConverterError):
            self.converter.pydantic_to_function_schema(dict)  # Not a Pydantic model


if __name__ == "__main__":
    unittest.main()
if __name__ == "__main__":
    unittest.main()
