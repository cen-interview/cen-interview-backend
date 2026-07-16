"""Alembic 환경 설정."""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from interview.api.database import Base
from interview.api.auth import model as auth_model
from interview.api.evidence import model as evidence_model
from interview.api.evidence import question_pattern_model
from interview.api.interviews import model as interviews_model
from interview.api.users import model as users_model
from interview.config import settings

if context.config.config_file_name is not None:
    fileConfig(context.config.config_file_name)

target_metadata = Base.metadata
context.config.set_main_option("sqlalchemy.url", settings.database_url)
_ = (auth_model, evidence_model, question_pattern_model, interviews_model, users_model)


def run_migrations_offline() -> None:
    context.configure(
        url=settings.database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        context.config.get_section(context.config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
