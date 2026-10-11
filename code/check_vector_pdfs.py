"""Check regenerated figures for raster content and embedded fonts."""
import json
from pathlib import Path
from pypdf import PdfReader
from pypdf.generic import ContentStream

ROOT = Path(__file__).resolve().parents[1]


def inspect(path):
    reader = PdfReader(path)
    images, inline_images, unembedded = [], [], []
    visited = set()

    def walk(resources, content, context):
        if content is not None:
            stream = ContentStream(content, reader)
            inline_images.extend(context for _, op in stream.operations if op == b"INLINE IMAGE")
        if resources is None:
            return
        resources = resources.get_object()
        for name, ref in resources.get("/Font", {}).items():
            font = ref.get_object()
            descendants = font.get("/DescendantFonts", [font])
            for item in descendants:
                desc = item.get_object().get("/FontDescriptor")
                if desc is None or not any(k in desc.get_object()
                                           for k in ["/FontFile", "/FontFile2", "/FontFile3"]):
                    unembedded.append(f"{context}:{name}")
        for name, ref in resources.get("/XObject", {}).items():
            obj = ref.get_object()
            key = (getattr(ref, "idnum", id(obj)), getattr(ref, "generation", 0))
            if key in visited:
                continue
            visited.add(key)
            if obj.get("/Subtype") == "/Image":
                images.append(f"{context}:{name}")
            elif obj.get("/Subtype") == "/Form":
                walk(obj.get("/Resources", resources), obj, f"{context}:{name}")

    for i, page in enumerate(reader.pages):
        walk(page.get("/Resources"), page.get_contents(), f"page{i + 1}")
    result = {"file": str(path.relative_to(ROOT)), "pages": len(reader.pages),
              "image_objects": images, "inline_images": inline_images,
              "unembedded_fonts": unembedded}
    if images or inline_images or unembedded:
        raise AssertionError(result)
    return result


if __name__ == "__main__":
    result = [inspect(path) for path in sorted((ROOT / "figures").rglob("*.pdf"))]
    (ROOT / "build").mkdir(exist_ok=True)
    (ROOT / "build" / "vector_pdf_check.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
