import asyncio
import logging

from sqlalchemy import text

from synapse.db.session import session_factory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run() -> None:
    logger.info("System 2 worker ready")
    while True:
        async with session_factory() as session:
            await session.execute(text("SELECT 1"))
        await asyncio.sleep(10)


if __name__ == "__main__":
    asyncio.run(run())
