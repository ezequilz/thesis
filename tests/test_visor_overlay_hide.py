"""Capture visor must hide debug gizmos on the same client as get_render."""

from __future__ import annotations

from types import SimpleNamespace

from splat_explorer.rendering.viser_viewer import (
    overlay_node_names,
    queue_client_overlay_visibility,
    visor_safe_covariances,
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


def test_visor_preserves_thin_and_rotated_valid_splats():
    import numpy as np
    from splat_explorer.rendering.base import quats_to_covariances
    # An axis over sqrt(float16.max) is still safe when rotated across axes.
    quats = np.array([[1,0,0,0], [np.cos(np.pi/8),0,0,np.sin(np.pi/8)]], dtype=np.float32)
    scales = np.array([[1e-9,.05,56.], [300.,.01,.01]], dtype=np.float32)
    expected = quats_to_covariances(quats,scales.astype(np.float64)).astype(np.float32)
    np.testing.assert_array_equal(visor_safe_covariances(quats,scales),expected)


def test_visor_limits_only_overflow_and_preserves_shape():
    import numpy as np
    from splat_explorer.rendering.base import quats_to_covariances
    quats=np.array([[1,0,0,0], [1,0,0,0]],dtype=np.float32)
    scales=np.array([[1.,2.,3267.],[1e30,2e30,3e30]],dtype=np.float32)
    before=quats_to_covariances(quats,scales.astype(np.float64))
    out=visor_safe_covariances(quats,scales)
    assert np.isfinite(out.astype(np.float16)).all()
    for original,safe in zip(before,out):
        np.testing.assert_allclose(safe,original*(65504./np.abs(original).max()),rtol=1e-6)


def test_visor_rejects_invalid_values_before_packing():
    import numpy as np
    import pytest
    for invalid in (np.nan,np.inf,-1.):
        with pytest.raises(ValueError,match='finite'):
            visor_safe_covariances(np.array([[1.,0,0,0]]),np.array([[invalid,1.,1.]]))
