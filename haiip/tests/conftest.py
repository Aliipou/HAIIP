"""Pytest configuration and shared fixtures for all tests.

Fixtures are scoped appropriately:
- session scope: DB engine, app (expensive to create)
- function scope: DB session (rolled back after each test), HTTP client

Security: tests use a dedicated in-memory SQLite DB — never touches production.
"""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from haiip.api.auth import create_access_token, hash_password
from haiip.api.database import Base, get_db
from haiip.api.main import create_app
from haiip.api.models import Tenant, User

# ── Test database ─────────────────────────────────────────────────────────────
TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture(scope="session")
async def test_engine():
    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture(scope="session")
def session_factory(test_engine):
    return async_sessionmaker(
        bind=test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )


@pytest_asyncio.fixture
async def db_session(session_factory):
    """Function-scoped session — rolled back after each test."""
    async with session_factory() as session:
        yield session
        await session.rollback()


# ── FastAPI test app ──────────────────────────────────────────────────────────

@pytest_asyncio.fixture(scope="session")
async def app(test_engine, session_factory):
    """Create test app with overridden DB dependency."""
    application = create_app()

    async def _override_get_db():
        async with session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    application.dependency_overrides[get_db] = _override_get_db
    return application


@pytest_asyncio.fixture
async def client(app):
    """Async HTTP client for testing routes."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac


# ── Seed data ─────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def test_tenant(db_session: AsyncSession) -> Tenant:
    tenant = Tenant(name="Test SME", slug="test-sme")
    db_session.add(tenant)
    await db_session.flush()
    await db_session.refresh(tenant)
    return tenant


@pytest_asyncio.fixture
async def test_admin(db_session: AsyncSession, test_tenant: Tenant) -> User:
    user = User(
        tenant_id=test_tenant.id,
        email="admin@test-sme.com",
        hashed_password=hash_password("Admin123!"),
        full_name="Test Admin",
        role="admin",
    )
    db_session.add(user)
    await db_session.flush()
    await db_session.refresh(user)
    return user


@pytest_asyncio.fixture
async def test_operator(db_session: AsyncSession, test_tenant: Tenant) -> User:
    user = User(
        tenant_id=test_tenant.id,
        email="operator@test-sme.com",
        hashed_password=hash_password("Operator123!"),
        full_name="Test Operator",
        role="operator",
    )
    db_session.add(user)
    await db_session.flush()
    await db_session.refresh(user)
    return user


@pytest_asyncio.fixture
def admin_token(test_admin: User, test_tenant: Tenant) -> str:
    return create_access_token(test_admin.id, test_tenant.id, test_admin.role)


@pytest_asyncio.fixture
def operator_token(test_operator: User, test_tenant: Tenant) -> str:
    return create_access_token(test_operator.id, test_tenant.id, test_operator.role)


@pytest_asyncio.fixture
def admin_headers(admin_token: str) -> dict:
    return {"Authorization": f"Bearer {admin_token}"}


@pytest_asyncio.fixture
def operator_headers(operator_token: str) -> dict:
    return {"Authorization": f"Bearer {operator_token}"}
