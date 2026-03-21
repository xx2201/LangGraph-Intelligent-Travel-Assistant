import pytest

from langgraph_study.backend.thread_store import (
    connect_thread_store,
    create_thread,
    default_thread_title,
    get_thread,
    initialize_thread_store,
    list_threads,
    update_thread_after_chat,
)


@pytest.mark.anyio
async def test_thread_store_create_and_list(tmp_path) -> None:
    db_path = tmp_path / "thread_store.sqlite"
    connection = await connect_thread_store(str(db_path))
    try:
        await initialize_thread_store(connection)
        created = await create_thread(connection, "thread-a")
        threads = await list_threads(connection)
    finally:
        await connection.close()

    assert created.thread_id == "thread-a"
    assert created.title == default_thread_title()
    assert len(threads) == 1
    assert threads[0].thread_id == "thread-a"


@pytest.mark.anyio
async def test_thread_store_update_after_chat(tmp_path) -> None:
    db_path = tmp_path / "thread_store.sqlite"
    connection = await connect_thread_store(str(db_path))
    try:
        await initialize_thread_store(connection)
        await create_thread(connection, "thread-b")
        updated = await update_thread_after_chat(
            connection,
            thread_id="thread-b",
            user_message="帮我做一个去成都的旅游规划",
            assistant_message="好的，我先给你一版成都行程建议。",
        )
        fetched = await get_thread(connection, "thread-b")
    finally:
        await connection.close()

    assert updated.title.startswith("帮我做一个去成都的旅游规划"[:5])
    assert updated.last_user_message == "帮我做一个去成都的旅游规划"
    assert fetched is not None
    assert fetched.last_assistant_message == "好的，我先给你一版成都行程建议。"
