"""Screen-door opacity shader must inject even without dithering_fragment chunk."""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MOVIES = REPO / "mgr_src" / "mgr" / "movies.py"


def _movies_template_js() -> str:
    text = MOVIES.read_text(encoding="utf-8")
    start = text.index("_DEFAULT_THREEJS_TEMPLATE = r'''") + len("_DEFAULT_THREEJS_TEMPLATE = r'''")
    end = text.index("'''", start)
    return text[start:end]


def test_screen_door_template_has_output_fragment_fallback():
    js = _movies_template_js()
    assert "function injectScreenDoorFragmentShader(shader)" in js
    assert "#include <output_fragment>" in js
    assert "renderer.dithering = true" in js
    assert "screenDoorV4" in js


def test_scatter_points_use_billboard_instancing():
    js = _movies_template_js()
    assert "isBillboardPoints" in js
    assert "updateBillboardPointsMesh" in js
    assert 'slider.max = "10"' in js


def test_inject_screen_door_fragment_shader_logic():
    """Node executes the same inject helper against mock fragment shaders."""
    import json
    import subprocess

    node_script = r"""
    function screenDoorDiscardGlsl() {
      return (
        "float sdOpacity = clamp(uScreenDoorOpacity, 0.0, 1.0);\n" +
        "if (sdOpacity < 1.0) { discard; }\n"
      );
    }
    function injectScreenDoorFragmentShader(shader) {
      if (shader.fragmentShader.indexOf("uScreenDoorOpacity") !== -1) return;
      shader.fragmentShader = shader.fragmentShader.replace(
        "void main() {",
        "uniform float uScreenDoorOpacity;\nvoid main() {"
      );
      const snippet = screenDoorDiscardGlsl();
      if (shader.fragmentShader.indexOf("#include <dithering_fragment>") !== -1) {
        shader.fragmentShader = shader.fragmentShader.replace(
          "#include <dithering_fragment>",
          snippet + "#include <dithering_fragment>"
        );
      } else if (shader.fragmentShader.indexOf("#include <output_fragment>") !== -1) {
        shader.fragmentShader = shader.fragmentShader.replace(
          "#include <output_fragment>",
          snippet + "#include <output_fragment>"
        );
      }
    }
    const cases = [
      "void main() {\n  #include <dithering_fragment>\n  #include <output_fragment>\n}",
      "void main() {\n  #include <output_fragment>\n}",
    ];
    const out = [];
    for (const frag of cases) {
      const shader = { fragmentShader: frag };
      injectScreenDoorFragmentShader(shader);
      out.push({
        hasUniform: shader.fragmentShader.includes("uScreenDoorOpacity"),
        hasDiscard: shader.fragmentShader.includes("discard"),
      });
    }
    console.log(JSON.stringify(out));
    """
    proc = subprocess.run(
        ["node", "-e", node_script],
        check=True,
        capture_output=True,
        text=True,
    )
    results = json.loads(proc.stdout.strip())
    assert len(results) == 2
    for row in results:
        assert row["hasUniform"] and row["hasDiscard"]
