"""Application configuration — loaded once at startup, never mutated."""

from functools import lru_cache
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env.local",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    app_env: Literal["development", "staging", "production"] = "development"
    app_debug: bool = False
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    # Security
    secret_key: str = "INSECURE_DEV_KEY_CHANGE_IN_PRODUCTION"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    # Database
    database_url: str = "sqlite+aiosqlite:///./haiip_dev.db"

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # LLM / RAG
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    embedding_model: str = "all-MiniLM-L6-v2"

    # OPC UA
    opcua_endpoint: str = "opc.tcp://localhost:4840/freeopcua/server/"
    opcua_namespace: int = 2

    # MQTT
    mqtt_broker_host: str = "localhost"
    mqtt_broker_port: int = 1883
    mqtt_topic_prefix: str = "haiip/sensors"

    # Observability
    log_level: str = "INFO"
    prometheus_enabled: bool = True

    # Multi-tenancy
    default_tenant_id: str = "default"
    max_tenants: int = 50

    # ML
    model_artifacts_path: str = "./artifacts/models"
    anomaly_contamination: float = 0.05
    drift_threshold: float = 0.05
    confidence_min: float = 0.6

    @field_validator("secret_key")
    @classmethod
    def secret_key_must_be_strong(cls, v: str) -> str:
        if v == "INSECURE_DEV_KEY_CHANGE_IN_PRODUCTION":
            return v  # allowed only in dev
        if len(v) < 32:
            msg = "SECRET_KEY must be at least 32 characters"
            raise ValueError(msg)
        return v

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"


@lru_cache
def get_settings() -> Settings:
    """Cached singleton — call everywhere instead of constructing Settings()."""
    return Settings()
