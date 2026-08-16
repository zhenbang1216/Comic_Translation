import argparse
from pathlib import Path

from comic_codex.datasets.baseline import (
    BaselineSpec,
    build_baseline_selection,
    write_baseline_selection,
)
from comic_codex.datasets.manifest import DatasetManifest

LOW_SHARPNESS = [
    ("data2/archive/train/2268363-22_jpg.rf.c648482be70f6121f748ad7ee5b63204.jpg", 2299.54),
    ("data2/archive/train/2333338-16_jpg.rf.96be4bccb5521322162c569802ae64ee.jpg", 3709.57),
    ("data2/archive/valid/31883-12_jpg.rf.3c4dd55a9ca2e7d0fd4cae5309d7df02.jpg", 4015.68),
    ("data2/archive/train/2381588-22_jpeg.rf.262588d78e5f04b9403e3d929c6a6eb4.jpg", 4058.86),
    ("data2/archive/train/2317371-15_png.rf.2bfc74c75f4ebed5cc27d80a47d53d09.jpg", 4092.46),
    ("data2/archive/train/788360-14_jpg.rf.32e6eade7aaab5d0890bead712415f8a.jpg", 4193.62),
    ("data2/archive/valid/2482524-14_jpg.rf.ad9e276888767ef6215bb7f6b7e3328d.jpg", 4350.82),
    ("data2/archive/train/788368-1_jpg.rf.32b8b65cde9e157ff58289c33ba8a02c.jpg", 4352.17),
    ("data2/archive/train/2302584-17_jpg.rf.bc6e6d94ccedcd87c948fafa365ea245.jpg", 4411.24),
    ("data2/archive/train/2380937-11_jpg.rf.dde2c48299db06c5267446ab2d2a9358.jpg", 4629.20),
    ("data2/archive/train/2322938-16_jpg.rf.ec63f40c36246a3e4fb631cf396944b1.jpg", 4650.91),
    ("data2/archive/valid/31819-22_jpg.rf.e88221fa318bb55b928b0a76b10abd97.jpg", 4707.16),
]

HIGH_SHARPNESS = [
    ("data2/archive/test/788350-16_jpg.rf.7f529df89e2b0110bbe2eb065c266d1d.jpg", 19008.68),
    ("data2/archive/train/40046-9_jpg.rf.4407a02f9ef79bb6031dc2deb716f955.jpg", 17547.39),
    ("data2/archive/train/788342-12_jpg.rf.a865cc5da523117ff77e5fc38ef014f0.jpg", 17532.36),
    ("data2/archive/test/40369-10_jpg.rf.4e3937134c5c50e05a3cb4ab2d0b7c60.jpg", 17039.32),
    ("data2/archive/test/40204-9_jpg.rf.2c1f61b3b05ec36a4ed6307340b0026f.jpg", 16918.50),
    ("data2/archive/valid/40360-13_jpg.rf.8040f533dd5d027849e4af29cbf5b5e0.jpg", 16731.25),
    ("data2/archive/train/40050-3_jpg.rf.888dfba25fc0880be7b6045b474fdf95.jpg", 16627.35),
    ("data2/archive/train/40381-12_jpg.rf.73bba057a53cd6481df3c873e6ab581f.jpg", 16609.20),
    ("data2/archive/valid/40061-10_jpg.rf.6a7759e50ff5be2bce3e48fb0cb66d68.jpg", 16554.35),
    ("data2/archive/train/788353-6_jpg.rf.f9e06425c3bf22e55ef1f3db26167139.jpg", 16274.22),
    ("data2/archive/valid/39903-7_jpg.rf.845e3b88b7e2a9f087d71743a17c084d.jpg", 16221.43),
    ("data2/archive/valid/788347-7_jpg.rf.7ce85481fc07c90d03432d683d76cd5a.jpg", 16068.55),
]

MANGA_ROOT = "data3/Manga109_released_2026_05_21 2/Manga109_released_2026_05_21"


def _data2_annotation(path: str) -> str:
    split = path.split("/")[2]
    return f"data2/archive/{split}/_annotations.coco.json"


def build_specs() -> list[BaselineSpec]:
    specs = [
        BaselineSpec(
            category="blurred_low_resolution",
            source_relative_path=path,
            annotation_relative_path=_data2_annotation(path),
            language="id",
            license_status="unknown",
            notes=f"512x512 low-sharpness candidate; Laplacian variance={score:.2f}",
        )
        for path, score in LOW_SHARPNESS
    ]
    specs.extend(
        BaselineSpec(
            category="clear_horizontal",
            source_relative_path=path,
            annotation_relative_path=_data2_annotation(path),
            language="id",
            license_status="unknown",
            notes=f"512x512 high-sharpness horizontal candidate; Laplacian variance={score:.2f}",
        )
        for path, score in HIGH_SHARPNESS
    )
    specs.extend(
        BaselineSpec(
            category="clear_vertical",
            source_relative_path=f"{MANGA_ROOT}/images/ARMS/{page:03d}.jpg",
            annotation_relative_path=f"{MANGA_ROOT}/annotations/ARMS.xml",
            language="ja",
            license_status="research_only",
            notes=(
                "Manga109 high-resolution Japanese vertical-text candidate; "
                "visual review required"
            ),
        )
        for page in range(3, 15)
    )
    specs.extend(
        BaselineSpec(
            category="complex_background",
            source_relative_path=f"{MANGA_ROOT}/images/ARMS/{page:03d}.jpg",
            annotation_relative_path=f"{MANGA_ROOT}/annotations/ARMS.xml",
            language="ja",
            license_status="research_only",
            notes="Manga109 high-detail background candidate; visual review required",
        )
        for page in range(15, 27)
    )
    return specs


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the 48-page baseline selection")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = DatasetManifest.model_validate_json(args.manifest.read_text(encoding="utf-8"))
    selection = build_baseline_selection(manifest, build_specs())
    write_baseline_selection(selection, args.output)
    print(f"records={len(selection.records)} output={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
