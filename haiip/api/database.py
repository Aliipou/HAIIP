"""Async SQLAlchemy engine + session factory.

Single source of truth for DB access. Import `AsyncSession` and `get_db`
everywhere — never construct sessions manually outside this module.
"""

from collections.abc import AsyncGenerator

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from haiip.api.config import get_settings

settings = get_settings()

# ── Engine ────────────────────────────────────────────────────────────────────
# SQLite gets WAL mode for better concurrency; Postgres ignores this event.
engine = create_async_engine(
    settings.database_url,
    echo=settings.app_debug,
    pool_pre_ping=True,
    # SQLite-specific: disable check_same_thread
    connect_args=(
        {"check_same_thread": False}
        if "sqlite" in settings.database_url
        else {}
    ),
)


@event.listens_for(engine.sync_engine, "connect")
def _set_sqlite_pragma(dbapi_conn, _connection_record):  # type: ignore[no-untyped-def]
    """Enable WAL mode and foreign keys for SQLite."""
    if "sqlite" in settings.database_url:
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


# ── Session factory ───────────────────────────────────────────────────────────
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


# ── Base model ────────────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    """All ORM models must inherit from this."""


# ── Dependency ────────────────────────────────────────────────────────────────
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency — yields a session, always closes on exit."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ── Lifecycle helpers ─────────────────────────────────────────────────────────
async def create_all_tables() -> None:
    """Create all tables — used in tests and dev startup."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def drop_all_tables() -> None:
    """Drop all tables — used in tests only."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


async def check_database_connection() -> bool:
    """Health check — returns True if DB is reachable."""
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
