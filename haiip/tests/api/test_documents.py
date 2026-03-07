"""Tests for document/RAG routes."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_query_no_documents(client: AsyncClient, operator_headers):
    response = await client.post(
        "/api/v1/query",
        json={"question": "What causes HDF failure?"},
        headers=operator_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "No documents" in data["answer"]
    assert data["confidence"] == 0.0


@pytest.mark.asyncio
async def test_query_requires_auth(client: AsyncClient):
    response = await client.post(
        "/api/v1/query",
        json={"question": "test question"},
    )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_query_too_short(client: AsyncClient, operator_headers):
    response = await client.post(
        "/api/v1/query",
        json={"question": "hi"},  # less than 3 chars
        headers=operator_headers,
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_ingest_document_engineer_only(client: AsyncClient, operator_headers):
    response = await client.post(
        "/api/v1/documents/ingest",
        json={"title": "Test Doc", "content": "This is test content for HAIIP."},
        headers=operator_headers,
    )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_ingest_document_success(client: AsyncClient, admin_headers):
    response = await client.post(
        "/api/v1/documents/ingest",
        json={
            "title": "CNC Maintenance Manual",
            "content": "HDF failure occurs when heat dissipation is insufficient. "
            "Check cooling system regularly to prevent thermal overload.",
            "source": "manual",
        },
        headers=admin_headers,
    )
    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "ingested"
    assert data["chunks_created"] >= 1


@pytest.mark.asyncio
async def test_ingest_empty_content_rejected(client: AsyncClient, admin_headers):
    response = await client.post(
        "/api/v1/documents/ingest",
        json={"title": "Bad Doc", "content": ""},
        headers=admin_headers,
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_ingest_missing_title_rejected(client: AsyncClient, admin_headers):
    response = await client.post(
        "/api/v1/documents/ingest",
        json={"content": "Some content without title"},
        headers=admin_headers,
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_document_stats(client: AsyncClient, operator_headers):
    response = await client.get("/api/v1/documents/stats", headers=operator_headers)
    assert response.status_code == 200
    data = response.json()
    assert "total_documents" in data
    assert "index_ready" in data
