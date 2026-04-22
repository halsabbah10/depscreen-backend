"""Route tests for /api/chat — conversation CRUD.

LLM-calling routes (send message, stream, auto-title) are deliberately
skipped at route level; they're covered at the service level in
test_services.py via mocked OpenAI clients. What we exercise here is
the conversation lifecycle: create → list → rename → get messages →
archive. Plus the RBAC guard that patients own their own conversations.
"""

from __future__ import annotations

from uuid import uuid4

# ─────────────────────────────────────────────────────────────────────────────
# Create / list
# ─────────────────────────────────────────────────────────────────────────────


def test_create_conversation_requires_patient_role(client, clinician_headers):
    resp = client.post(
        "/api/chat/conversations",
        headers=clinician_headers,
        json={"title": "Test", "context_type": "general"},
    )
    assert resp.status_code == 403


def test_patient_creates_conversation(client, patient_headers):
    resp = client.post(
        "/api/chat/conversations",
        headers=patient_headers,
        json={"title": "Feeling off today", "context_type": "general"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["title"] == "Feeling off today"
    assert data["context_type"] == "general"
    assert data["message_count"] == 0
    assert data["is_active"] is True


def test_list_conversations_returns_own_only(client, patient_user, patient_headers, db):
    """Patient A shouldn't see Patient B's conversations."""
    from app.models.db import Conversation, User

    # Seed a conversation for patient_user
    db.add(
        Conversation(
            id=str(uuid4()),
            user_id=patient_user.id,
            title="My convo",
            context_type="general",
            is_active=True,
        )
    )
    # A conversation owned by a different patient
    other = User(
        id=str(uuid4()),
        email="other@test.local",
        password_hash="x",
        full_name="Other",
        role="patient",
        is_active=True,
    )
    db.add(other)
    db.flush()
    db.add(
        Conversation(
            id=str(uuid4()),
            user_id=other.id,
            title="Someone else's convo",
            context_type="general",
            is_active=True,
        )
    )
    db.commit()

    resp = client.get("/api/chat/conversations", headers=patient_headers)
    assert resp.status_code == 200
    titles = [c["title"] for c in resp.json()]
    assert "My convo" in titles
    assert "Someone else's convo" not in titles


# ─────────────────────────────────────────────────────────────────────────────
# Rename
# ─────────────────────────────────────────────────────────────────────────────


def test_rename_conversation(client, patient_headers):
    create = client.post(
        "/api/chat/conversations",
        headers=patient_headers,
        json={"title": "Old title", "context_type": "general"},
    )
    conv_id = create.json()["id"]

    rename = client.patch(
        f"/api/chat/conversations/{conv_id}",
        headers=patient_headers,
        json={"title": "A quieter week"},
    )
    assert rename.status_code == 200
    assert rename.json()["title"] == "A quieter week"

    # Confirm it persisted
    lst = client.get("/api/chat/conversations", headers=patient_headers)
    matching = [c for c in lst.json() if c["id"] == conv_id]
    assert matching[0]["title"] == "A quieter week"


def test_rename_rejects_empty_title(client, patient_headers):
    create = client.post(
        "/api/chat/conversations",
        headers=patient_headers,
        json={"title": "Some title", "context_type": "general"},
    )
    conv_id = create.json()["id"]

    resp = client.patch(
        f"/api/chat/conversations/{conv_id}",
        headers=patient_headers,
        json={"title": "   "},
    )
    assert resp.status_code == 400


def test_rename_rejects_overlong_title(client, patient_headers):
    create = client.post(
        "/api/chat/conversations",
        headers=patient_headers,
        json={"title": "Some title", "context_type": "general"},
    )
    conv_id = create.json()["id"]

    resp = client.patch(
        f"/api/chat/conversations/{conv_id}",
        headers=patient_headers,
        json={"title": "x" * 150},
    )
    assert resp.status_code == 400


def test_rename_404_on_other_users_conversation(client, patient_headers, db):
    from app.models.db import Conversation, User

    other = User(
        id=str(uuid4()),
        email="stranger@test.local",
        password_hash="x",
        full_name="Stranger",
        role="patient",
        is_active=True,
    )
    db.add(other)
    db.flush()
    theirs = Conversation(
        id=str(uuid4()),
        user_id=other.id,
        title="Theirs",
        context_type="general",
        is_active=True,
    )
    db.add(theirs)
    db.commit()

    resp = client.patch(
        f"/api/chat/conversations/{theirs.id}",
        headers=patient_headers,
        json={"title": "Hacked"},
    )
    assert resp.status_code == 404


# ─────────────────────────────────────────────────────────────────────────────
# Archive
# ─────────────────────────────────────────────────────────────────────────────


def test_archive_conversation_soft_deletes(client, patient_headers, db):
    from app.models.db import Conversation

    create = client.post(
        "/api/chat/conversations",
        headers=patient_headers,
        json={"title": "To archive", "context_type": "general"},
    )
    conv_id = create.json()["id"]

    resp = client.delete(f"/api/chat/conversations/{conv_id}", headers=patient_headers)
    assert resp.status_code == 200

    # Row still exists but is_active = False
    row = db.query(Conversation).filter(Conversation.id == conv_id).first()
    assert row is not None
    assert row.is_active is False

    # List no longer includes it (filters by is_active)
    lst = client.get("/api/chat/conversations", headers=patient_headers)
    ids = [c["id"] for c in lst.json()]
    assert conv_id not in ids


# ─────────────────────────────────────────────────────────────────────────────
# Get messages
# ─────────────────────────────────────────────────────────────────────────────


def test_get_messages_returns_chronological(client, patient_user, patient_headers, db):
    from datetime import UTC, datetime, timedelta

    from app.models.db import ChatMessage, Conversation

    conv_id = str(uuid4())
    db.add(
        Conversation(
            id=conv_id,
            user_id=patient_user.id,
            title="Test",
            context_type="general",
            is_active=True,
        )
    )
    now = datetime.now(UTC)
    # Seed out-of-order so we can prove the endpoint sorts
    db.add(
        ChatMessage(
            id=str(uuid4()),
            conversation_id=conv_id,
            role="assistant",
            content="Second",
            created_at=now,
        )
    )
    db.add(
        ChatMessage(
            id=str(uuid4()),
            conversation_id=conv_id,
            role="user",
            content="First",
            created_at=now - timedelta(seconds=5),
        )
    )
    db.commit()

    resp = client.get(f"/api/chat/conversations/{conv_id}/messages", headers=patient_headers)
    assert resp.status_code == 200
    msgs = resp.json()["messages"]
    assert [m["content"] for m in msgs] == ["First", "Second"]


def test_get_messages_404_on_foreign_conversation(client, patient_headers, db):
    from app.models.db import Conversation, User

    other = User(
        id=str(uuid4()),
        email="foreign@test.local",
        password_hash="x",
        full_name="Foreign",
        role="patient",
        is_active=True,
    )
    db.add(other)
    db.flush()
    theirs = Conversation(
        id=str(uuid4()),
        user_id=other.id,
        title="Theirs",
        context_type="general",
        is_active=True,
    )
    db.add(theirs)
    db.commit()

    resp = client.get(f"/api/chat/conversations/{theirs.id}/messages", headers=patient_headers)
    assert resp.status_code == 404
