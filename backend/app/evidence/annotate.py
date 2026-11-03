"""Draw the violation on a copy of the (privacy-masked) frame for evidence."""

from __future__ import annotations

import cv2
import numpy as np

from app.ppe.types import Box

RED, YELLOW, WHITE, BLACK = (40, 40, 230), (0, 215, 255), (255, 255, 255), (0, 0, 0)


def _i(b: Box) -> tuple[int, int, int, int]:
    return int(b[0]), int(b[1]), int(b[2]), int(b[3])


def annotate_violation(image: np.ndarray, person_box: Box, head_roi: Box | None, label: str) -> np.ndarray:
    out = image.copy()
    x1, y1, x2, y2 = _i(person_box)
    cv2.rectangle(out, (x1, y1), (x2, y2), RED, 2)
    if head_roi is not None:
        hx1, hy1, hx2, hy2 = _i(head_roi)
        cv2.rectangle(out, (hx1, hy1), (hx2, hy2), YELLOW, 1)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    ty = y1 - 6 if y1 - th - 8 > 0 else y2 + th + 6  # keep the label off the person
    cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 6, ty + 4), BLACK, -1)
    cv2.putText(out, label, (x1 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    return out
