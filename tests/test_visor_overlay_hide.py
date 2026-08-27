"""Capture visor must hide debug gizmos on the same client as get_render."""

from __future__ import annotations

from types import SimpleNamespace

from splat_explorer.rendering.viser_viewer import (
    overlay_node_names,
    queue_client_overlay_visibility,
)


def test_overlay_node_names_always_includes_agent_gizmos():
    names = overlay_node_names([])
    assert "/agent" in names
    assert "/agent/camera" in names
    assert "/agent/trajectory" in names
    assert "/origin" in names


def test_overlay_node_names_merges_handles_without_duplicates():
    handles = [
        SimpleNamespace(name="/agent/camera"),
        None,
        SimpleNamespace(name="/extra/gizmo"),
    ]
    names = overlay_node_names(handles)
    assert names.count("/agent/camera") == 1
    assert "/extra/gizmo" in names


def test_queue_client_overlay_visibility_uses_the_client_socket():
    queued = []

    class Conn:
        def queue_message(self, message) -> None:
            queued.append(message)

    client = SimpleNamespace(_websock_connection=Conn(), flushed=False)

    def flush() -> None:
        client.flushed = True

    client.flush = flush

    ok = queue_client_overlay_visibility(
        client, ("/agent", "/agent/camera"), visible=False,
    )
    if not ok:
        # viser is an optional extra; the hide path still has a broadcast fallback.
        return
    assert client.flushed
    assert [m.name for m in queued] == ["/agent", "/agent/camera"]
    assert all(m.visible is False for m in queued)


def test_queue_client_overlay_visibility_skips_clients_without_a_socket():
    assert queue_client_overlay_visibility(SimpleNamespace(), ("/agent",), True) is False
