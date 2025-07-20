import pytest
from unittest import mock
from django.test import override_settings
from swirl.openai.openai import OpenAIClient, AI_QUERY_USE


@pytest.mark.django_db
def test_openai_client_custom_base_url():
    with override_settings(
        OPENAI_API_KEY='test-key',
        OPENAI_API_BASE='http://bedrock.aws',
        OPENAI_API_VERSION='2023-05-15',
        AZURE_OPENAI_KEY='',
        AZURE_MODEL='',
        AZURE_OPENAI_ENDPOINT='',
    ):
        with mock.patch('openai.OpenAI') as mock_openai:
            instance = mock.MagicMock()
            mock_openai.return_value = instance
            client = OpenAIClient(AI_QUERY_USE)
            mock_openai.assert_called_once_with(
                api_key='test-key',
                base_url='http://bedrock.aws',
                api_version='2023-05-15'
            )
            assert client.openai_client == instance

