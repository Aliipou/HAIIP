"""Tests for auth routes and JWT logic."""

import pytest
from httpx import AsyncClient

from haiip.api.auth import (
    TokenError,
    create_access_token,
    create_refresh_token,
    decode_access_token,
    decode_refresh_token,
    hash_password,
    verify_password,
)

# ── Password hashing ──────────────────────────────────────────────────────────


def test_hash_password_produces_different_hash():
    h1 = hash_password("secret123A")
    h2 = hash_password("secret123A")
    assert h1 != h2  # bcrypt uses unique salt each time


def test_verify_password_correct():
    hashed = hash_password("MyPassword1!")
    assert verify_password("MyPassword1!", hashed) is True


def test_verify_password_wrong():
    hashed = hash_password("MyPassword1!")
    assert verify_password("WrongPassword1!", hashed) is False


# ── Token creation / decoding ─────────────────────────────────────────────────


def test_access_token_decode_roundtrip():
    token = create_access_token("user-1", "tenant-1", "admin")
    payload = decode_access_token(token)
    assert payload["sub"] == "user-1"
    assert payload["tenant_id"] == "tenant-1"
    assert payload["role"] == "admin"
    assert payload["type"] == "access"


def test_refresh_token_decode_roundtrip():
    token = create_refresh_token("user-1", "tenant-1")
    payload = decode_refresh_token(token)
    assert payload["sub"] == "user-1"
    assert payload["tenant_id"] == "tenant-1"
    assert payload["type"] == "refresh"


def test_access_token_rejected_as_refresh():
    token = create_access_token("user-1", "tenant-1", "admin")
    with pytest.raises(TokenError, match="Expected token type"):
        decode_refresh_token(token)


def test_refresh_token_rejected_as_access():
    token = create_refresh_token("user-1", "tenant-1")
    with pytest.raises(TokenError, match="Expected token type"):
        decode_access_token(token)


def test_invalid_token_raises_token_error():
    with pytest.raises(TokenError):
        decode_access_token("not.a.valid.token")


def test_tampered_token_raises_token_error():
    token = create_access_token("user-1", "tenant-1", "admin")
    tampered = token[:-5] + "XXXXX"
    with pytest.raises(TokenError):
        decode_access_token(tampered)


# ── Auth routes ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_me_returns_current_user(client: AsyncClient, admin_headers, test_admin):
    response = await client.get("/api/v1/auth/me", headers=admin_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["email"] == test_admin.email
    assert data["role"] == "admin"
    assert "hashed_password" not in data


@pytest.mark.asyncio
async def test_get_me_requires_auth(client: AsyncClient):
    response = await client.get("/api/v1/auth/me")
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_get_me_invalid_token(client: AsyncClient):
    response = await client.get(
        "/api/v1/auth/me",
        headers={"Authorization": "Bearer invalid.token.here"},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_register_admin_only(client: AsyncClient, operator_headers):
    response = await client.post(
        "/api/v1/auth/register",
        json={
            "email": "new@test.com",
            "password": "NewUser123!",
            "full_name": "New User",
            "role": "operator",
        },
        headers=operator_headers,
    )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_register_new_user(client: AsyncClient, admin_headers):
    response = await client.post(
        "/api/v1/auth/register",
        json={
            "email": "newengineer@test.com",
            "password": "Engineer123!",
            "full_name": "New Engineer",
            "role": "engineer",
        },
        headers=admin_headers,
    )
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "newengineer@test.com"
    assert data["role"] == "engineer"
    assert "hashed_password" not in data


@pytest.mark.asyncio
async def test_register_weak_password_rejected(client: AsyncClient, admin_headers):
    response = await client.post(
        "/api/v1/auth/register",
        json={
            "email": "weak@test.com",
            "password": "weakpass",  # no uppercase, no digit
            "full_name": "Weak User",
        },
        headers=admin_headers,
    )
    assert response.status_code == 422
