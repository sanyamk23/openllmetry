"""Unit tests for Azure AI Search instrumentation.

Uses mock clients since Azure Search requires a cloud endpoint.
Spans are verified via the InMemorySpanExporter from conftest.py.
"""

from unittest.mock import MagicMock

import pytest
from opentelemetry.semconv_ai import SpanAttributes


def _make_search_client(index_name="test-index"):
    """Create a mock SearchClient with patched endpoint and index."""
    from azure.search.documents import SearchClient

    client = MagicMock(spec=SearchClient)
    client._endpoint = "https://test.search.windows.net"
    client._index_name = index_name
    return client


def _make_index_client():
    """Create a mock SearchIndexClient."""
    from azure.search.documents.indexes import SearchIndexClient

    client = MagicMock(spec=SearchIndexClient)
    client._endpoint = "https://test.search.windows.net"
    return client


def _make_indexer_client():
    """Create a mock SearchIndexerClient."""
    from azure.search.documents.indexes import SearchIndexerClient

    client = MagicMock(spec=SearchIndexerClient)
    client._endpoint = "https://test.search.windows.net"
    return client


# ---------------------------------------------------------------------------
# SearchClient tests
# ---------------------------------------------------------------------------


def test_search_creates_span(exporter):
    client = _make_search_client()
    client.search.return_value = MagicMock()

    client.search(search_text="hello world", top=5, filter="category eq 'docs'")

    spans = exporter.get_finished_spans()
    search_spans = [s for s in spans if s.name == "azure_search.search"]
    assert len(search_spans) == 1

    span = search_spans[0]
    assert span.attributes.get(SpanAttributes.VECTOR_DB_VENDOR) == "azure_search"
    assert span.attributes.get(SpanAttributes.AZURE_SEARCH_SEARCH_TEXT) == "hello world"
    assert span.attributes.get(SpanAttributes.AZURE_SEARCH_TOP) == 5
    assert span.attributes.get(SpanAttributes.AZURE_SEARCH_FILTER) == "category eq 'docs'"
    assert span.attributes.get(SpanAttributes.AZURE_SEARCH_INDEX_NAME) == "test-index"
    assert span.attributes.get("server.address") == "https://test.search.windows.net"


def test_get_document_creates_span(exporter):
    client = _make_search_client()
    client.get_document.return_value = {"id": "1", "title": "Test"}

    client.get_document(document_id="1")

    spans = exporter.get_finished_spans()
    get_doc_spans = [s for s in spans if s.name == "azure_search.get_document"]
    assert len(get_doc_spans) == 1

    span = get_doc_spans[0]
    assert span.attributes.get(SpanAttributes.AZURE_SEARCH_INDEX_NAME) == "test-index"


def test_autocomplete_creates_span(exporter):
    client = _make_search_client()
    client.autocomplete.return_value = MagicMock()

    client.autocomplete(search_text="hel", autocomplete_mode="twoTerms")

    spans = exporter.get_finished_spans()
    ac_spans = [s for s in spans if s.name == "azure_search.autocomplete"]
    assert len(ac_spans) == 1

    span = ac_spans[0]
    assert span.attributes.get(SpanAttributes.AZURE_SEARCH_AUTOCOMPLETE_TEXT) == "hel"


def test_suggest_creates_span(exporter):
    client = _make_search_client()
    client.suggest.return_value = MagicMock()

    client.suggest(search_text="hel", suggester_name="sg")

    spans = exporter.get_finished_spans()
    suggest_spans = [s for s in spans if s.name == "azure_search.suggest"]
    assert len(suggest_spans) == 1

    span = suggest_spans[0]
    assert span.attributes.get(SpanAttributes.AZURE_SEARCH_SUGGEST_TEXT) == "hel"


def test_index_documents_creates_span(exporter):
    client = _make_search_client()
    response = MagicMock()
    response.results = [MagicMock(error=None), MagicMock(error=None)]
    client.index_documents.return_value = response

    docs = [{"id": "1", "title": "A"}, {"id": "2", "title": "B"}]
    client.index_documents(documents=docs)

    spans = exporter.get_finished_spans()
    idx_spans = [s for s in spans if s.name == "azure_search.index_documents"]
    assert len(idx_spans) == 1

    span = idx_spans[0]
    assert span.attributes.get(SpanAttributes.AZURE_SEARCH_DOCUMENTS_COUNT) == 2
    assert span.attributes.get(SpanAttributes.AZURE_SEARCH_SUCCEEDED_COUNT) == 2


def test_upload_documents_creates_span(exporter):
    client = _make_search_client()
    response = MagicMock()
    response.results = [MagicMock(error=None)]
    client.upload_documents.return_value = response

    client.upload_documents(documents=[{"id": "1"}])

    spans = exporter.get_finished_spans()
    spans = [s for s in spans if s.name == "azure_search.upload_documents"]
    assert len(spans) == 1
    assert spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_DOCUMENTS_COUNT) == 1


def test_delete_documents_creates_span(exporter):
    client = _make_search_client()
    response = MagicMock()
    response.results = [
        MagicMock(error=None),
        MagicMock(error=None),
        MagicMock(error=None),
    ]
    client.delete_documents.return_value = response

    client.delete_documents(documents=[{"id": "1"}, {"id": "2"}, {"id": "3"}])

    spans = exporter.get_finished_spans()
    del_spans = [s for s in spans if s.name == "azure_search.delete_documents"]
    assert len(del_spans) == 1
    assert del_spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_DOCUMENTS_COUNT) == 3


def test_get_document_count_creates_span(exporter):
    client = _make_search_client()
    client.get_document_count.return_value = 42

    client.get_document_count()

    spans = exporter.get_finished_spans()
    count_spans = [s for s in spans if s.name == "azure_search.get_document_count"]
    assert len(count_spans) == 1
    assert count_spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_DOCUMENTS_COUNT) == 42


# ---------------------------------------------------------------------------
# SearchIndexClient tests
# ---------------------------------------------------------------------------


def test_create_index_creates_span(exporter):
    client = _make_index_client()
    index = MagicMock()
    index.name = "my-index"
    client.create_index.return_value = index

    client.create_index(index=index)

    spans = exporter.get_finished_spans()
    create_spans = [s for s in spans if s.name == "azure_search.create_index"]
    assert len(create_spans) == 1
    assert create_spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_INDEX_NAME) == "my-index"


def test_delete_index_creates_span(exporter):
    client = _make_index_client()
    client.delete_index.return_value = None

    client.delete_index(index_name="my-index")

    spans = exporter.get_finished_spans()
    del_spans = [s for s in spans if s.name == "azure_search.delete_index"]
    assert len(del_spans) == 1
    assert del_spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_INDEX_NAME) == "my-index"


def test_get_index_creates_span(exporter):
    client = _make_index_client()
    index = MagicMock()
    index.name = "my-index"
    client.get_index.return_value = index

    client.get_index(index_name="my-index")

    spans = exporter.get_finished_spans()
    get_spans = [s for s in spans if s.name == "azure_search.get_index"]
    assert len(get_spans) == 1
    assert get_spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_INDEX_NAME) == "my-index"


def test_get_index_statistics_creates_span(exporter):
    client = _make_index_client()
    stats = MagicMock()
    stats.document_count = 1000
    stats.storage_size = 1048576
    client.get_index_statistics.return_value = stats

    client.get_index_statistics(index_name="my-index")

    spans = exporter.get_finished_spans()
    stat_spans = [s for s in spans if s.name == "azure_search.get_index_statistics"]
    assert len(stat_spans) == 1
    assert stat_spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_INDEX_DOC_COUNT) == 1000
    assert stat_spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_INDEX_SIZE_BYTES) == 1048576


def test_get_service_statistics_creates_span(exporter):
    client = _make_index_client()
    counters = MagicMock()
    counters.search_service_usage = "85%"
    counters.search_service_limit = "100%"
    stats = MagicMock()
    stats.counters = counters
    client.get_service_statistics.return_value = stats

    client.get_service_statistics()

    spans = exporter.get_finished_spans()
    svc_spans = [s for s in spans if s.name == "azure_search.get_service_statistics"]
    assert len(svc_spans) == 1
    assert svc_spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_SERVICE_USAGE) == "85%"
    assert svc_spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_SERVICE_LIMIT) == "100%"


# ---------------------------------------------------------------------------
# SearchIndexerClient tests
# ---------------------------------------------------------------------------


def test_create_indexer_creates_span(exporter):
    client = _make_indexer_client()
    indexer = MagicMock()
    indexer.name = "my-indexer"
    client.create_indexer.return_value = indexer

    client.create_indexer(indexer=indexer)

    spans = exporter.get_finished_spans()
    idx_spans = [s for s in spans if s.name == "azure_search.create_indexer"]
    assert len(idx_spans) == 1
    assert idx_spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_INDEXER_NAME) == "my-indexer"


def test_get_indexer_status_creates_span(exporter):
    client = _make_indexer_client()
    status = MagicMock()
    status.status = "running"
    client.get_indexer_status.return_value = status

    client.get_indexer_status(indexer_name="my-indexer")

    spans = exporter.get_finished_spans()
    status_spans = [s for s in spans if s.name == "azure_search.get_indexer_status"]
    assert len(status_spans) == 1
    assert status_spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_INDEXER_STATUS) == "running"


def test_run_indexer_creates_span(exporter):
    client = _make_indexer_client()
    client.run_indexer.return_value = None

    client.run_indexer(indexer_name="my-indexer")

    spans = exporter.get_finished_spans()
    run_spans = [s for s in spans if s.name == "azure_search.run_indexer"]
    assert len(run_spans) == 1
    assert run_spans[0].attributes.get(SpanAttributes.AZURE_SEARCH_INDEXER_NAME) == "my-indexer"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_search_exception_records_exception(exporter):
    client = _make_search_client()
    client.search.side_effect = RuntimeError("Connection refused")

    with pytest.raises(RuntimeError, match="Connection refused"):
        client.search(search_text="fail")

    spans = exporter.get_finished_spans()
    search_spans = [s for s in spans if s.name == "azure_search.search"]
    assert len(search_spans) == 1

    span = search_spans[0]
    assert span.status.status_code.name == "ERROR"
    assert len(span.events) == 1
    assert span.events[0].name == "exception"


# ---------------------------------------------------------------------------
# Instrumentor API
# ---------------------------------------------------------------------------


def test_instrumentor_has_correct_interface():
    from opentelemetry.instrumentation.azure_search import AzureSearchInstrumentor

    instrumentor = AzureSearchInstrumentor()
    assert hasattr(instrumentor, "_instrument")
    assert hasattr(instrumentor, "_uninstrument")
    assert "azure-search-documents" in instrumentor.instrumentation_dependencies()[0]
