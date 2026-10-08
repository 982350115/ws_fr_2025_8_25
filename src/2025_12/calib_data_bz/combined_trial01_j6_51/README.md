# Combined checkerboard captures

This directory copies 45 samples from trial01 and 6 from trial02_j6/trial02_j6_02. Original sessions, trial01 split, and baseline transform are unchanged. samples.json is a 51-record read-only composite, not a new 30/15 split.

merge_manifest.json records source JSON checksums and verifies every copied image. board_stability_report.json compares board poses using the old samples as a fixed reference. Its conclusion is limited by robot geometry, camera mount, and PnP errors.
